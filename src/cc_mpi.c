#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE
#define _XOPEN_SOURCE 700

#include "cc_mpi.h"

#include <mpi.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static int default_num_threads(void)
{
    long t = sysconf(_SC_NPROCESSORS_ONLN);
    return (t > 0) ? (int)t : 1;
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

/* sort compare */
static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t*)a;
    uint32_t y = *(const uint32_t*)b;
    return (x<y)?-1:(x>y)?1:0;
}

/* lower_bound on sorted uint32_t array */
static uint32_t lower_bound_u32(const uint32_t *arr, uint32_t n, uint32_t key, int *found)
{
    uint32_t lo=0, hi=n;
    while (lo<hi)
    {
        uint32_t mid = lo + (hi-lo)/2;
        uint32_t v = arr[mid];
        if (v < key) lo = mid+1;
        else hi = mid;
    }
    if (found) *found = (lo < n && arr[lo] == key);
    return lo;
}

/* compare BoundaryEdge */
static int cmp_be(const void *a, const void *b)
{
    const BoundaryEdge *x=(const BoundaryEdge*)a;
    const BoundaryEdge *y=(const BoundaryEdge*)b;
    if (x->rep != y->rep) return (x->rep<y->rep)?-1:1;
    if (x->remote != y->remote) return (x->remote<y->remote)?-1:1;
    return 0;
}

/* compare BoundaryPair */
static int cmp_bp(const void *a, const void *b)
{
    const BoundaryPair *x=(const BoundaryPair*)a;
    const BoundaryPair *y=(const BoundaryPair*)b;
    if (x->rep != y->rep) return (x->rep<y->rep)?-1:1;
    if (x->gidx != y->gidx) return (x->gidx<y->gidx)?-1:1;
    return 0;
}

/* rebuild boundary pairs (rep,gidx) from boundary edges (rep,remote) */
static void rebuild_boundary_pairs(BPVec *pairs,
                                   const BEVec *edges,
                                   const uint32_t *ghost_vertices,
                                   uint32_t ghost_count,
                                   MPI_Comm comm)
{
    pairs->size = 0;

    if (edges->size == 0 || ghost_count == 0) return;

    for (uint64_t i=0; i<edges->size; ++i)
    {
        uint32_t rep = edges->data[i].rep;
        uint32_t rv  = edges->data[i].remote;

        int found=0;
        uint32_t gidx = lower_bound_u32(ghost_vertices, ghost_count, rv, &found);
        if (!found) continue; /* if missing, it'll be added later via ghost expansion */
        bpvec_push(pairs, (BoundaryPair){rep, gidx}, comm);
    }

    if (pairs->size > 1)
    {
        qsort(pairs->data, (size_t)pairs->size, sizeof(BoundaryPair), cmp_bp);

        uint64_t w=0;
        for (uint64_t i=0; i<pairs->size; ++i)
        {
            if (i==0 ||
                pairs->data[i].rep  != pairs->data[w-1].rep ||
                pairs->data[i].gidx != pairs->data[w-1].gidx)
                pairs->data[w++] = pairs->data[i];
        }
        pairs->size = w;
    }
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

    int found=0;
    uint32_t gidx = lower_bound_u32(ghost_vertices, ghost_count, x, &found);
    if (found)
    {
        if (missing) *missing = 0;
        return ghost_labels[gidx];
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

    U32Vec ghosts_tmp; u32vec_init(&ghosts_tmp);
    BEVec boundary_edges; bevec_init(&boundary_edges);

    /* count local-only edges + collect remote neighbors as initial ghosts */
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
            else if (v < n_global)
                u32vec_push(&ghosts_tmp, v, comm);
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

        compute_connected_components_pthreads(&Glocal, comp_of, num_threads, chunk_size);
    }

    free(lrow); free(lcol);
    Glocal.row_ptr = NULL;
    Glocal.col_idx = NULL;

    /* component rep list */
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

    /* build boundary edges using comp_of */
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
                bevec_push(&boundary_edges, (BoundaryEdge){rep, v}, comm);
            }
        }

        if (boundary_edges.size > 1)
        {
            qsort(boundary_edges.data, (size_t)boundary_edges.size, sizeof(BoundaryEdge), cmp_be);

            uint64_t w=0;
            for (uint64_t i=0; i<boundary_edges.size; ++i)
            {
                if (i==0 ||
                    boundary_edges.data[i].rep    != boundary_edges.data[w-1].rep ||
                    boundary_edges.data[i].remote != boundary_edges.data[w-1].remote)
                    boundary_edges.data[w++] = boundary_edges.data[i];
            }
            boundary_edges.size = w;
        }
    }

    /* ---------------- Ghost set: sort+unique ghosts_tmp into ghost_vertices ---------------- */

    uint32_t *ghost_vertices = NULL;
    uint32_t ghost_count = 0;

    if (ghosts_tmp.size > 0)
    {
        qsort(ghosts_tmp.data, (size_t)ghosts_tmp.size, sizeof(uint32_t), cmp_u32);

        uint64_t w=0;
        for (uint64_t i=0; i<ghosts_tmp.size; ++i)
        {
            if (i==0 || ghosts_tmp.data[i] != ghosts_tmp.data[w-1])
                ghosts_tmp.data[w++] = ghosts_tmp.data[i];
        }

        if (w > (uint64_t)UINT32_MAX) die_abort(comm, "Too many ghosts");
        ghost_count = (uint32_t)w;
        ghost_vertices = ghosts_tmp.data;
        ghosts_tmp.data = NULL;
        ghosts_tmp.size = ghosts_tmp.cap = 0;
    }
    u32vec_free(&ghosts_tmp);

    uint32_t *ghost_labels = NULL;
    if (ghost_count > 0)
    {
        ghost_labels = (uint32_t*)malloc((size_t)ghost_count * sizeof(uint32_t));
        if (!ghost_labels) die_abort(comm, "OOM: ghost_labels");
        for (uint32_t i=0; i<ghost_count; ++i) ghost_labels[i] = ghost_vertices[i];
    }

    BPVec boundary_pairs; bpvec_init(&boundary_pairs);
    rebuild_boundary_pairs(&boundary_pairs, &boundary_edges, ghost_vertices, ghost_count, comm);

    /* ---------------- Phase 2: SV-ish merge (hook + pointer-jumping) ---------------- */

    bool global_changed = false;
    int merge_rounds = 0;

    if (n_local > 0 && ghost_count > 0 && boundary_pairs.size > 0)
        global_changed = true;

    ExchangePlan plan;
    memset(&plan, 0, sizeof(plan));
    if (ghost_count > 0)
        exchangeplan_build(&plan, ghost_vertices, ghost_count, n_global, comm, owner_of_vertex);

    while (global_changed)
    {
        /* (A) Pack outgoing labels for vertices other ranks request from us */
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

        /* (B) Exchange; updates ghost_labels */
        exchangeplan_exchange(&plan, ghost_labels, comm);

        /* (C) Hooking */
        bool local_changed = false;

        for (int k=0; k<rep_list.size; ++k)
        {
            uint32_t rep = rep_list.data[k];
            comp_min[rep] = comp_label[rep];
        }

        for (uint64_t e=0; e<boundary_pairs.size; ++e)
        {
            uint32_t rep  = boundary_pairs.data[e].rep;
            uint32_t gidx = boundary_pairs.data[e].gidx;
            uint32_t nlbl = ghost_labels[gidx];
            if (nlbl < comp_min[rep]) comp_min[rep] = nlbl;
        }

        /* (D) Pointer jumping with dynamic “parent ghost” discovery */
        const int MAX_JUMPS = 4;

        U32Vec new_parents; u32vec_init(&new_parents);
        bool need_rebuild = false;

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
                if (missing)
                {
                    if (!(x >= v_start && x < v_end) && x < n_global)
                    {
                        u32vec_push(&new_parents, x, comm);
                        need_rebuild = true;
                    }
                    break;
                }
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

        /* Expand ghost set and rebuild plan/pairs if needed */
        if (need_rebuild && new_parents.size > 0)
        {
            qsort(new_parents.data, (size_t)new_parents.size, sizeof(uint32_t), cmp_u32);

            uint64_t w=0;
            for (uint64_t i=0; i<new_parents.size; ++i)
            {
                if (i==0 || new_parents.data[i] != new_parents.data[w-1])
                    new_parents.data[w++] = new_parents.data[i];
            }
            new_parents.size = w;

            uint32_t merged_cap = ghost_count + (uint32_t)new_parents.size;
            uint32_t *merged = (uint32_t*)malloc((size_t)merged_cap * sizeof(uint32_t));
            if (!merged) die_abort(comm, "OOM: merged ghosts");

            uint32_t i=0, j=0, m=0;
            while (i<ghost_count && j<(uint32_t)new_parents.size)
            {
                uint32_t a = ghost_vertices[i];
                uint32_t b = new_parents.data[j];
                if (a < b) merged[m++] = a, i++;
                else if (b < a) merged[m++] = b, j++;
                else merged[m++] = a, i++, j++;
            }
            while (i<ghost_count) merged[m++] = ghost_vertices[i++];
            while (j<(uint32_t)new_parents.size) merged[m++] = new_parents.data[j++];

            uint32_t *new_labels = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
            if (!new_labels) die_abort(comm, "OOM: new ghost_labels");

            for (uint32_t t=0; t<m; ++t) new_labels[t] = merged[t];

            for (uint32_t old=0; old<ghost_count; ++old)
            {
                uint32_t v = ghost_vertices[old];
                int found=0;
                uint32_t idx = lower_bound_u32(merged, m, v, &found);
                if (found) new_labels[idx] = ghost_labels[old];
            }

            free(ghost_vertices);
            free(ghost_labels);

            ghost_vertices = merged;
            ghost_labels   = new_labels;
            ghost_count    = m;

            exchangeplan_free(&plan);
            exchangeplan_build(&plan, ghost_vertices, ghost_count, n_global, comm, owner_of_vertex);

            boundary_pairs.size = 0;
            rebuild_boundary_pairs(&boundary_pairs, &boundary_edges, ghost_vertices, ghost_count, comm);

            local_changed = true;
        }

        u32vec_free(&new_parents);

        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, comm);
        ++merge_rounds;
    }

    if (comm_rank == 0)
        fprintf(stderr, "CC merge(SV hook+jump) converged in %d rounds\n", merge_rounds);

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

    bpvec_free(&boundary_pairs);
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
