#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE
#define _XOPEN_SOURCE 700

#include "cc_mpi.h"

#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "cc.h"              /* compute_connected_components_pthreads() */
#include "exchange_plan.h"
#include "vec_helpers.h"

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

/* ---------------- utility ---------------- */

static void die_abort(MPI_Comm comm, const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(comm, EXIT_FAILURE);
}

/* Use OpenMP thread setting as the pthread count to avoid oversubscription. */
static int default_num_threads(void)
{
    int t = omp_get_max_threads(); /* respects OMP_NUM_THREADS */
    return (t > 0) ? t : 1;
}

/* Must match the same block partition as your DistCSRGraph loader */
static int owner_of_vertex(uint32_t v, uint32_t n_global, int comm_size)
{
    if (comm_size <= 0) return 0;

    uint32_t cs = (uint32_t)comm_size;
    uint32_t base = (cs > 0) ? (n_global / cs) : 0;
    uint32_t rem  = (cs > 0) ? (n_global % cs) : 0;

    uint64_t threshold = (uint64_t)(base + 1u) * (uint64_t)rem;
    if ((uint64_t)v < threshold)
    {
        uint32_t denom = base + 1u;
        return denom ? (int)(v / denom) : 0;
    }
    if (base == 0) return (int)rem;

    uint64_t shifted = (uint64_t)v - threshold;
    return (int)(rem + (uint32_t)(shifted / base));
}

/* compare BoundaryEdge by (remote, rep) so all remotes group together */
static int cmp_be_remote_first(const void *a, const void *b)
{
    const BoundaryEdge *x = (const BoundaryEdge*)a;
    const BoundaryEdge *y = (const BoundaryEdge*)b;
    if (x->remote != y->remote) return (x->remote < y->remote) ? -1 : 1;
    if (x->rep    != y->rep)    return (x->rep    < y->rep)    ? -1 : 1;
    return 0;
}

/* get label of a GLOBAL vertex x:
   - if x is local: label is comp_label[rep(x)]
   - if x is ghost: label is ghost_labels[gidx(x)]
   - otherwise: missing=1
*/
static inline uint32_t get_vertex_label_global(uint32_t x,
                                               const uint32_t *comp_label,
                                               const uint32_t *comp_of,
                                               uint32_t v_start, uint32_t v_end,
                                               const uint32_t *ghost_vertices, uint32_t ghost_count,
                                               const uint32_t *ghost_labels,
                                               int *missing)
{
    if (x >= v_start && x < v_end)
    {
        uint32_t li = x - v_start;
        uint32_t rep = comp_of[li];
        if (missing) *missing = 0;
        return comp_label[rep];
    }

    /* NOTE: in this version, we DO NOT do dynamic ghost expansion,
       so any "parent" jump should always be either local or already in ghost list.
       We'll do binary search only for pointer-jump. */
    uint32_t lo=0, hi=ghost_count;
    while (lo < hi)
    {
        uint32_t mid = lo + (hi-lo)/2;
        uint32_t v = ghost_vertices[mid];
        if (v < x) lo = mid+1;
        else hi = mid;
    }
    if (lo < ghost_count && ghost_vertices[lo] == x)
    {
        if (missing) *missing = 0;
        return ghost_labels[lo];
    }

    if (missing) *missing = 1;
    return x;
}

void compute_connected_components_mpi_advanced(const DistCSRGraph *restrict Gd,
                                               uint32_t *restrict labels_global,
                                               int chunk_size,
                                               int exchange_interval,
                                               MPI_Comm comm)
{
    (void)exchange_interval;

    int comm_rank=0, comm_size=0;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    const uint32_t n_global = Gd->n_global;
    const uint32_t v_start  = Gd->v_start;
    const uint32_t v_end    = Gd->v_end;
    const uint32_t n_local  = Gd->n_local;

    const uint64_t *row_ptr = Gd->row_ptr;
    const uint32_t *col_idx = Gd->col_idx;

    if (n_global == 0) return;

    /* ---------------- Phase 1: induced local-only graph + pthread CC ---------------- */

    uint64_t *lrow = (uint64_t*)calloc((size_t)n_local + 1, sizeof(uint64_t));
    if (!lrow && n_local>0) die_abort(comm, "OOM: lrow");

    /* count local-only edges */
    for (uint32_t li=0; li<n_local; ++li)
    {
        uint64_t begin = row_ptr[li];
        uint64_t end   = row_ptr[li+1];

        uint64_t cnt_local = 0;
        for (uint64_t j=begin; j<end; ++j)
        {
            uint32_t v = col_idx[j];
            if (v >= v_start && v < v_end)
                cnt_local++;
        }
        lrow[li+1] = lrow[li] + cnt_local;
    }

    uint64_t m_local_local = (n_local>0) ? lrow[n_local] : 0;
    uint32_t *lcol = NULL;
    if (m_local_local > 0)
    {
        lcol = (uint32_t*)malloc((size_t)m_local_local * sizeof(uint32_t));
        if (!lcol) die_abort(comm, "OOM: lcol");
    }

    for (uint32_t li=0; li<n_local; ++li)
    {
        uint64_t write = lrow[li];
        uint64_t begin = row_ptr[li];
        uint64_t end   = row_ptr[li+1];

        for (uint64_t j=begin; j<end; ++j)
        {
            uint32_t v = col_idx[j];
            if (v >= v_start && v < v_end)
                lcol[write++] = (uint32_t)(v - v_start);
        }
    }

    CSRGraph Glocal;
    memset(&Glocal, 0, sizeof(Glocal));
    Glocal.n = n_local;
    Glocal.m = m_local_local;
    Glocal.row_ptr = lrow;
    Glocal.col_idx = lcol;

    uint32_t *comp_of = NULL; /* local vertex -> local representative (min local id) */
    if (n_local > 0)
    {
        comp_of = (uint32_t*)malloc((size_t)n_local * sizeof(uint32_t));
        if (!comp_of) die_abort(comm, "OOM: comp_of");

        int num_threads = default_num_threads();
        if (num_threads < 1) num_threads = 1;

        double t_cc0 = MPI_Wtime();
        compute_connected_components_pthreads(&Glocal, comp_of, num_threads, chunk_size);
        double t_cc1 = MPI_Wtime();

        printf("[rank %d] Local CC done: n_local=%u m_local_local=%llu threads=%d time=%.3fs\n",
                comm_rank, n_local, (unsigned long long)m_local_local, num_threads, t_cc1 - t_cc0);
    }

    free(lrow); free(lcol);
    Glocal.row_ptr = NULL;
    Glocal.col_idx = NULL;

    /* rep list + labels */
    uint8_t *is_rep = NULL;
    U32VecI rep_list; u32veci_init(&rep_list);

    uint32_t *comp_label = NULL; /* indexed by local vertex id; valid for reps */
    uint32_t *comp_min   = NULL;

    if (n_local > 0)
    {
        is_rep = (uint8_t*)calloc((size_t)n_local, sizeof(uint8_t));
        if (!is_rep) die_abort(comm, "OOM: is_rep");

        comp_label = (uint32_t*)malloc((size_t)n_local * sizeof(uint32_t));
        comp_min   = (uint32_t*)malloc((size_t)n_local * sizeof(uint32_t));
        if (!comp_label || !comp_min) die_abort(comm, "OOM: comp_label/comp_min");

        for (uint32_t i=0; i<n_local; ++i) comp_label[i] = v_start + i;

        for (uint32_t li=0; li<n_local; ++li)
        {
            uint32_t r = comp_of[li];
            if (r < n_local && !is_rep[r])
            {
                is_rep[r] = 1;
                u32veci_push(&rep_list, r, comm);
            }
        }
    }

    /* ---------------- Boundary edges (ONLY) ----------------
       We DO NOT build ghosts_tmp.
       We will build ghost_vertices from boundary_edges later in linear time. */
    double t_b0 = MPI_Wtime();

    BEVec boundary_edges; bevec_init(&boundary_edges);

    if (n_local > 0)
    {
        for (uint32_t li=0; li<n_local; ++li)
        {
            uint32_t rep = comp_of[li];

            uint64_t begin = row_ptr[li];
            uint64_t end   = row_ptr[li+1];

            for (uint64_t j=begin; j<end; ++j)
            {
                uint32_t v = col_idx[j];
                if (v >= v_start && v < v_end) continue;
                if (v >= n_global) continue;
                /* store (rep, remote_global) */
                bevec_push(&boundary_edges, (BoundaryEdge){rep, v}, comm);
            }
        }

        if (boundary_edges.size > 1)
        {
            qsort(boundary_edges.data, (size_t)boundary_edges.size, sizeof(BoundaryEdge), cmp_be_remote_first);

            /* dedup identical (remote,rep) */
            uint64_t w=0;
            for (uint64_t i=0; i<boundary_edges.size; ++i)
            {
                if (i==0 ||
                    boundary_edges.data[i].remote != boundary_edges.data[w-1].remote ||
                    boundary_edges.data[i].rep    != boundary_edges.data[w-1].rep)
                {
                    boundary_edges.data[w++] = boundary_edges.data[i];
                }
            }
            boundary_edges.size = w;
        }
    }

    double t_b1 = MPI_Wtime();
    printf("[rank %d] boundary_edges=%llu build_boundary=%.3fs\n",
            comm_rank, (unsigned long long)boundary_edges.size, t_b1 - t_b0);

    /* ---------------- Build ghost list + map in ONE LINEAR PASS ---------------- */

    double t_g0 = MPI_Wtime();

    uint32_t ghost_count = 0;
    uint32_t *ghost_vertices = NULL;
    uint32_t *ghost_labels   = NULL;

    if (boundary_edges.size > 0)
    {
        /* worst-case: each boundary edge has unique remote -> allocate that */
        ghost_vertices = (uint32_t*)malloc((size_t)boundary_edges.size * sizeof(uint32_t));
        if (!ghost_vertices) die_abort(comm, "OOM: ghost_vertices");

        uint32_t prev = UINT32_MAX;

        for (uint64_t e=0; e<boundary_edges.size; ++e)
        {
            uint32_t rv = boundary_edges.data[e].remote; /* remote global */
            if (rv != prev)
            {
                ghost_vertices[ghost_count++] = rv;
                prev = rv;
            }
            /* overwrite remote field with ghost index */
            boundary_edges.data[e].remote = ghost_count - 1;
        }

        /* shrink to fit */
        uint32_t *shrunk = (uint32_t*)realloc(ghost_vertices, (size_t)ghost_count * sizeof(uint32_t));
        if (shrunk) ghost_vertices = shrunk;

        ghost_labels = (uint32_t*)malloc((size_t)ghost_count * sizeof(uint32_t));
        if (!ghost_labels) die_abort(comm, "OOM: ghost_labels");
        for (uint32_t i=0; i<ghost_count; ++i) ghost_labels[i] = ghost_vertices[i];
    }

    double t_g1 = MPI_Wtime();

    printf("[rank %d] ghost_count=%u build_ghosts+map=%.3fs\n",
            comm_rank, ghost_count, t_g1 - t_g0);

    /* ---------------- Phase 2: SV-ish merge (hook + pointer-jumping) ---------------- */

    bool global_changed = false;
    int merge_rounds = 0;

    if (n_local > 0 && ghost_count > 0 && boundary_edges.size > 0)
        global_changed = true;

    ExchangePlan plan;
    memset(&plan, 0, sizeof(plan));
    if (ghost_count > 0)
        exchangeplan_build(&plan, ghost_vertices, ghost_count, n_global, comm, owner_of_vertex);

    printf("[rank %d] plan built indegree=%d outdegree=%d send=%d recv=%d\n",
            comm_rank, plan.indegree, plan.outdegree, plan.total_send_to, plan.total_need_from);

    if (comm_rank == 1)
    {
        printf("[rank 1] DEBUG: ghost_count=%u boundary_edges=%llu reps=%d\n",
                ghost_count, (unsigned long long)boundary_edges.size, rep_list.size);
    }

    /* Make iteration-1 exchange timing reflect comm, not "waiting for last rank" */
    MPI_Barrier(comm);

    const int MAX_JUMPS = 4;

    while (global_changed)
    {
        double it0 = MPI_Wtime();

        /* (A) Pack outgoing labels for vertices other ranks request from us */
        double t_pack0 = MPI_Wtime();
        if (plan.total_send_to > 0)
        {
            for (int i=0; i<plan.total_send_to; ++i)
            {
                uint32_t v = plan.send_to_flat[i];
                uint32_t lbl = v;

                if (v >= v_start && v < v_end)
                {
                    uint32_t li  = v - v_start;
                    uint32_t rep = comp_of[li];
                    lbl = comp_label[rep];
                }
                plan.send_labels_flat[i] = lbl;
            }
        }
        double t_pack1 = MPI_Wtime();

        /* (B) Exchange; updates ghost_labels */
        double t_ex0 = MPI_Wtime();
        exchangeplan_exchange(&plan, ghost_labels, comm);
        double t_ex1 = MPI_Wtime();

        /* (C) Hooking */
        double t_hook0 = MPI_Wtime();
        bool local_changed = false;

        for (int k=0; k<rep_list.size; ++k)
        {
            uint32_t rep = rep_list.data[k];
            comp_min[rep] = comp_label[rep];
        }

        for (uint64_t e=0; e<boundary_edges.size; ++e)
        {
            uint32_t rep  = boundary_edges.data[e].rep;
            uint32_t gidx = boundary_edges.data[e].remote; /* now ghost index */
            uint32_t nlbl = ghost_labels[gidx];
            if (nlbl < comp_min[rep]) comp_min[rep] = nlbl;
        }
        double t_hook1 = MPI_Wtime();

        /* (D) Pointer jumping (no dynamic ghost expansion in this version) */
        double t_jump0 = MPI_Wtime();

        for (int k=0; k<rep_list.size; ++k)
        {
            uint32_t rep = rep_list.data[k];

            uint32_t x = comp_min[rep];
            for (int t=0; t<MAX_JUMPS; ++t)
            {
                int missing=0;
                uint32_t y = get_vertex_label_global(x,
                                                     comp_label, comp_of,
                                                     v_start, v_end,
                                                     ghost_vertices, ghost_count,
                                                     ghost_labels,
                                                     &missing);
                if (missing) break;
                if (y >= x) break;
                x = y;
            }

            if (x < comp_min[rep]) comp_min[rep] = x;
        }

        for (int k=0; k<rep_list.size; ++k)
        {
            uint32_t rep = rep_list.data[k];
            if (comp_min[rep] < comp_label[rep])
            {
                comp_label[rep] = comp_min[rep];
                local_changed = true;
            }
        }

        double t_jump1 = MPI_Wtime();

        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, comm);
        ++merge_rounds;

        double it1 = MPI_Wtime();

        if (merge_rounds <= 3 || (merge_rounds % 10 == 0))
        {
            printf("[rank %d] iter=%d pack=%.3fs exch=%.3fs hook=%.3fs jump=%.3fs total=%.3fs changed=%d\n",
                    comm_rank, merge_rounds,
                    t_pack1 - t_pack0,
                    t_ex1   - t_ex0,
                    t_hook1 - t_hook0,
                    t_jump1 - t_jump0,
                    it1 - it0,
                    (int)local_changed);
        }
    }

    if (comm_rank == 0)
        printf("CC merge(hook+jump) converged in %d rounds\n", merge_rounds);
    /* ---------------- Final per-vertex labels + allgather ---------------- */

    uint32_t *local_out = NULL;
    if (n_local > 0)
    {
        local_out = (uint32_t*)malloc((size_t)n_local * sizeof(uint32_t));
        if (!local_out) die_abort(comm, "OOM: local_out");
        for (uint32_t li=0; li<n_local; ++li)
        {
            uint32_t rep = comp_of[li];
            local_out[li] = comp_label[rep];
        }
    }

    uint32_t base = (comm_size > 0) ? (n_global / (uint32_t)comm_size) : 0;
    uint32_t rem  = (comm_size > 0) ? (n_global % (uint32_t)comm_size) : 0;

    int *all_counts = (int*)malloc((size_t)comm_size * sizeof(int));
    int *all_displs = (int*)malloc((size_t)comm_size * sizeof(int));
    if (!all_counts || !all_displs) die_abort(comm, "OOM: all_counts/all_displs");

    int off=0;
    for (int r=0; r<comm_size; ++r)
    {
        uint32_t ln = ((uint32_t)r < rem) ? (base + 1u) : base;
        all_counts[r] = (int)ln;
        all_displs[r] = off;
        off += (int)ln;
    }

    MPI_Allgatherv(local_out, (int)n_local, MPI_UINT32_T,
                   labels_global, all_counts, all_displs, MPI_UINT32_T,
                   comm);

    /* ---------------- Cleanup ---------------- */

    free(all_counts);
    free(all_displs);
    free(local_out);

    exchangeplan_free(&plan);

    bevec_free(&boundary_edges);

    free(ghost_vertices);
    free(ghost_labels);

    free(comp_of);
    free(comp_label);
    free(comp_min);

    free(is_rep);
    u32veci_free(&rep_list);
}

/* unchanged helper */
uint32_t count_connected_components(const uint32_t *restrict labels_global, uint32_t n_global)
{
    if (n_global == 0 || labels_global == NULL) return 0;

    unsigned char *comp_bitmap = (unsigned char*)calloc((size_t)n_global, sizeof(unsigned char));
    if (!comp_bitmap)
    {
        fprintf(stderr, "count_connected_components: Failed to allocate bitmap\n");
        return 0;
    }

    uint32_t comp_count = 0;
    for (uint32_t v=0; v<n_global; ++v)
    {
        uint32_t rep = labels_global[v];
        if (rep >= n_global) continue;
        if (comp_bitmap[rep] == 0)
        {
            comp_bitmap[rep] = 1;
            comp_count++;
        }
    }

    free(comp_bitmap);
    return comp_count;
}
