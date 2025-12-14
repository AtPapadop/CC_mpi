#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE

#include "cc_mpi.h"

#include <mpi.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#include "graph.h"
#include "cc.h"
#include "exchange_plan.h"
#include "vec_helpers.h"

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif

/* ============================================================
   Utilities / config
   ============================================================ */

static int g_cc_mpi_threads = 0;

void cc_mpi_set_num_threads(int nthreads)
{
    g_cc_mpi_threads = (nthreads > 0) ? nthreads : 0;
}

static void die_abort(MPI_Comm comm, const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(comm, EXIT_FAILURE);
}

static int default_num_threads(void)
{
#ifdef _SC_NPROCESSORS_ONLN
    long t = sysconf(_SC_NPROCESSORS_ONLN);
    return (t > 0) ? (int)t : 1;
#else
    return 1;
#endif
}

/* Must match DistCSRGraph partitioning (contiguous blocks) */
static int owner_of_vertex(uint32_t v, uint32_t n_global, int comm_size)
{
    if (comm_size <= 1) return 0;

    uint32_t cs   = (uint32_t)comm_size;
    uint32_t base = n_global / cs;
    uint32_t rem  = n_global % cs;

    uint64_t split = (uint64_t)(base + 1u) * (uint64_t)rem;
    if ((uint64_t)v < split)
        return (int)(v / (base + 1u));

    if (base == 0) return (int)rem; /* degenerate */
    return (int)(rem + (uint32_t)(((uint64_t)v - split) / base));
}

/* ============================================================
   Fast in-place sort: American-flag radix sort on key(remote,rep)
   key = (remote<<32) | rep
   ============================================================ */

static inline uint64_t be_key(const BoundaryEdge *e)
{
    return (((uint64_t)e->remote) << 32) | (uint64_t)e->rep;
}

static inline unsigned be_digit(const BoundaryEdge *e, int shift)
{
    return (unsigned)((be_key(e) >> shift) & 0xFFu);
}

static void be_insertion_sort(BoundaryEdge *a, uint64_t n)
{
    for (uint64_t i = 1; i < n; ++i)
    {
        BoundaryEdge x = a[i];
        uint64_t kx = be_key(&x);
        uint64_t j = i;
        while (j > 0)
        {
            uint64_t kj = be_key(&a[j - 1]);
            if (kj <= kx) break;
            a[j] = a[j - 1];
            --j;
        }
        a[j] = x;
    }
}

/* In-place American-flag sort by bytes, MSD first.
   Uses O(256) extra memory; no full-size temp buffer. */
static void be_afsort_rec(BoundaryEdge *a, uint64_t n, int shift)
{
    if (n <= 64 || shift < 0)
    {
        be_insertion_sort(a, n);
        return;
    }

    uint32_t count[256];
    uint32_t start[256];
    uint32_t next[256];

    memset(count, 0, sizeof(count));

    for (uint64_t i = 0; i < n; ++i)
        count[be_digit(&a[i], shift)]++;

    uint32_t sum = 0;
    for (int b = 0; b < 256; ++b)
    {
        start[b] = sum;
        sum += count[b];
        next[b] = start[b];
    }

    /* in-place permutation into buckets */
    for (int b = 0; b < 256; ++b)
    {
        uint32_t end = start[b] + count[b];
        while (next[b] < end)
        {
            uint32_t i = next[b];
            unsigned d = be_digit(&a[i], shift);
            if ((int)d == b)
            {
                next[b]++;
            }
            else
            {
                uint32_t j = next[d]++;
                BoundaryEdge tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }

    /* recurse buckets */
    if (shift == 0) return;
    int nshift = shift - 8;

    for (int b = 0; b < 256; ++b)
    {
        uint32_t c = count[b];
        if (c <= 1) continue;
        be_afsort_rec(a + start[b], (uint64_t)c, nshift);
    }
}

static void be_afsort(BoundaryEdge *a, uint64_t n)
{
    /* key is 64-bit; sort from top byte (shift=56) */
    be_afsort_rec(a, n, 56);
}

/* ============================================================
   get label of global vertex x (for pointer-jump)
   Requires ghost_vertices sorted.
   ============================================================ */
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
        uint32_t li  = x - v_start;
        uint32_t rep = comp_of[li];
        if (missing) *missing = 0;
        return comp_label[rep];
    }

    /* lower_bound on sorted ghost_vertices */
    uint32_t lo = 0, hi = ghost_count;
    while (lo < hi)
    {
        uint32_t mid = lo + (hi - lo) / 2;
        uint32_t v = ghost_vertices[mid];
        if (v < x) lo = mid + 1;
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

/* ============================================================
   Merge thread pool (pthreads)
   We do two phases per iteration:
     PH_INIT: comp_min[r] = comp_label[r]
     PH_UPD : pointer-jump + apply comp_label from comp_min
   Hooking across boundary edges remains serial (race-free, streaming)
   ============================================================ */

enum { PH_IDLE = 0, PH_INIT = 1, PH_UPD = 2, PH_STOP = 3 };

typedef struct
{
    int tid;
    int nthreads;

    const uint32_t *rep_list;
    uint32_t rep_count;

    uint32_t *comp_label;
    uint32_t *comp_min;

    /* for pointer-jump */
    const uint32_t *comp_of;
    uint32_t v_start, v_end;

    const uint32_t *ghost_vertices;
    uint32_t ghost_count;
    const uint32_t *ghost_labels;

    volatile int *phase;
    volatile int *changed;
    pthread_barrier_t *bar;
} MergeWorker;

static void *merge_worker_main(void *arg)
{
    MergeWorker *W = (MergeWorker*)arg;

    while (1)
    {
        pthread_barrier_wait(W->bar);

        int ph = *W->phase;
        if (ph == PH_STOP) break;

        uint32_t reps = W->rep_count;
        uint32_t start = (uint32_t)(((uint64_t)reps * (uint64_t)W->tid) / (uint64_t)W->nthreads);
        uint32_t end   = (uint32_t)(((uint64_t)reps * (uint64_t)(W->tid + 1)) / (uint64_t)W->nthreads);

        if (ph == PH_INIT)
        {
            for (uint32_t i = start; i < end; ++i)
            {
                uint32_t r = W->rep_list[i];
                W->comp_min[r] = W->comp_label[r];
            }
        }
        else if (ph == PH_UPD)
        {
            int local_any = 0;
            const int MAX_JUMPS = 4;

            for (uint32_t i = start; i < end; ++i)
            {
                uint32_t r = W->rep_list[i];

                uint32_t x = W->comp_min[r];
                for (int t = 0; t < MAX_JUMPS; ++t)
                {
                    int missing = 0;
                    uint32_t y = get_vertex_label_global(x,
                                                         W->comp_label, W->comp_of,
                                                         W->v_start, W->v_end,
                                                         W->ghost_vertices, W->ghost_count,
                                                         W->ghost_labels,
                                                         &missing);
                    if (missing) break;
                    if (y >= x) break;
                    x = y;
                }

                if (x < W->comp_min[r]) W->comp_min[r] = x;

                if (W->comp_min[r] < W->comp_label[r])
                {
                    W->comp_label[r] = W->comp_min[r];
                    local_any = 1;
                }
            }

            if (local_any) *W->changed = 1;
        }

        pthread_barrier_wait(W->bar);
    }

    pthread_barrier_wait(W->bar);
    return NULL;
}

/* ============================================================
   Main algorithm
   ============================================================ */

void compute_connected_components_mpi_advanced(const DistCSRGraph *restrict Gd,
                                               uint32_t *restrict labels_global,
                                               int chunk_size,
                                               int exchange_interval,
                                               MPI_Comm comm)
{
    (void)labels_global;
    if (exchange_interval < 1) exchange_interval = 1;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const uint32_t n_global = Gd->n_global;
    const uint32_t v_start  = Gd->v_start;
    const uint32_t v_end    = Gd->v_end;
    const uint32_t n_local  = Gd->n_local;

    const uint64_t *row_ptr = Gd->row_ptr;
    const uint32_t *col_idx = Gd->col_idx;

    if (n_global == 0 || n_local == 0) return;

    double t_total0 = MPI_Wtime();

    /* ============================================================
       Pass 1: count local-only edges for induced graph AND count boundary edges
       ============================================================ */
    uint64_t *lrow = (uint64_t*)calloc((size_t)n_local + 1, sizeof(uint64_t));
    if (!lrow) die_abort(comm, "OOM: lrow");

    uint64_t boundary_raw = 0;

    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint64_t begin = row_ptr[li];
        uint64_t end   = row_ptr[li + 1];

        uint64_t cnt_local = 0;
        for (uint64_t j = begin; j < end; ++j)
        {
            uint32_t v = col_idx[j];
            if (v >= v_start && v < v_end) cnt_local++;
            else if (v < n_global) boundary_raw++;
        }
        lrow[li + 1] = lrow[li] + cnt_local;
    }

    uint64_t m_local_local = lrow[n_local];

    /* build local-only col array */
    uint32_t *lcol = NULL;
    if (m_local_local > 0)
    {
        lcol = (uint32_t*)malloc((size_t)m_local_local * sizeof(uint32_t));
        if (!lcol) die_abort(comm, "OOM: lcol");
    }

    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint64_t write = lrow[li];
        uint64_t begin = row_ptr[li];
        uint64_t end   = row_ptr[li + 1];

        for (uint64_t j = begin; j < end; ++j)
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

    /* ============================================================
       Phase 1: local CC (pthreads)
       ============================================================ */
    uint32_t *comp_of = (uint32_t*)malloc((size_t)n_local * sizeof(uint32_t));
    if (!comp_of) die_abort(comm, "OOM: comp_of");

    int nthreads = (g_cc_mpi_threads > 0) ? g_cc_mpi_threads : default_num_threads();
    if (nthreads < 1) nthreads = 1;

    double t_cc0 = MPI_Wtime();
    compute_connected_components_pthreads(&Glocal, comp_of, nthreads, chunk_size);
    double t_cc1 = MPI_Wtime();

    printf("[rank %d] Local CC done: n_local=%u m_local=%llu threads=%d time=%.3fs\n",
           rank, n_local, (unsigned long long)m_local_local, nthreads, t_cc1 - t_cc0);

    free(lrow);
    free(lcol);
    Glocal.row_ptr = NULL;
    Glocal.col_idx = NULL;

    /* ============================================================
       Build rep_list without pushes/reallocs
       ============================================================ */
    double t_rep0 = MPI_Wtime();

    uint8_t *is_rep = (uint8_t*)calloc((size_t)n_local, 1);
    if (!is_rep) die_abort(comm, "OOM: is_rep");

    uint32_t rep_count = 0;
    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint32_t r = comp_of[li];
        if (r < n_local && !is_rep[r])
        {
            is_rep[r] = 1;
            rep_count++;
        }
    }

    uint32_t *rep_list = (uint32_t*)malloc((size_t)rep_count * sizeof(uint32_t));
    if (!rep_list) die_abort(comm, "OOM: rep_list");

    uint32_t wrep = 0;
    for (uint32_t r = 0; r < n_local; ++r)
        if (is_rep[r]) rep_list[wrep++] = r;

    double t_rep1 = MPI_Wtime();
    if (rank == 0) printf("[rank 0] reps=%u build_reps=%.3fs\n", rep_count, t_rep1 - t_rep0);

    /* labels for reps (stored at rep index) */
    uint32_t *comp_label = (uint32_t*)malloc((size_t)n_local * sizeof(uint32_t));
    uint32_t *comp_min   = (uint32_t*)malloc((size_t)n_local * sizeof(uint32_t));
    if (!comp_label || !comp_min) die_abort(comm, "OOM: comp_label/comp_min");

    for (uint32_t i = 0; i < n_local; ++i)
        comp_label[i] = v_start + i;

    /* ============================================================
       Build boundary edges into a single preallocated array (no vector growth)
       ============================================================ */
    double t_b0 = MPI_Wtime();

    uint64_t boundary_sz = boundary_raw;
    BoundaryEdge *boundary = NULL;

    if (boundary_sz > 0)
    {
        boundary = (BoundaryEdge*)malloc((size_t)boundary_sz * sizeof(BoundaryEdge));
        if (!boundary) die_abort(comm, "OOM: boundary edges");
    }

    uint64_t wb = 0;
    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint32_t rep = comp_of[li];

        uint64_t begin = row_ptr[li];
        uint64_t end   = row_ptr[li + 1];

        for (uint64_t j = begin; j < end; ++j)
        {
            uint32_t v = col_idx[j];
            if (v >= v_start && v < v_end) continue;
            if (v >= n_global) continue;
            boundary[wb++] = (BoundaryEdge){ rep, v };
        }
    }
    boundary_sz = wb;

    double t_b_build = MPI_Wtime();

    /* sort by (remote,rep) using fast in-place radix sort */
    if (boundary_sz > 1)
        be_afsort(boundary, boundary_sz);

    double t_b_sort = MPI_Wtime();

    /* dedup identical (remote,rep) */
    if (boundary_sz > 1)
    {
        uint64_t w = 0;
        for (uint64_t i = 0; i < boundary_sz; ++i)
        {
            if (i == 0 ||
                boundary[i].remote != boundary[w - 1].remote ||
                boundary[i].rep    != boundary[w - 1].rep)
            {
                boundary[w++] = boundary[i];
            }
        }
        boundary_sz = w;

        BoundaryEdge *shr = (BoundaryEdge*)realloc(boundary, (size_t)boundary_sz * sizeof(BoundaryEdge));
        if (shr) boundary = shr;
    }

    double t_b1 = MPI_Wtime();

    printf("[rank %d] boundary_edges raw=%llu dedup=%llu build=%.3fs sort=%.3fs dedup=%.3fs\n",
           rank,
           (unsigned long long)boundary_raw,
           (unsigned long long)boundary_sz,
           t_b_build - t_b0,
           t_b_sort  - t_b_build,
           t_b1      - t_b_sort);

    /* ============================================================
       Build ghost list (unique remote vertices) in one linear pass
       boundary is sorted by remote then rep.
       ============================================================ */
    double t_g0 = MPI_Wtime();

    uint32_t ghost_count = 0;
    uint32_t *ghost_vertices = NULL;
    uint32_t *ghost_labels   = NULL;

    if (boundary_sz > 0)
    {
        ghost_vertices = (uint32_t*)malloc((size_t)boundary_sz * sizeof(uint32_t));
        if (!ghost_vertices) die_abort(comm, "OOM: ghost_vertices");

        uint32_t prev = UINT32_MAX;
        for (uint64_t i = 0; i < boundary_sz; ++i)
        {
            uint32_t rv = boundary[i].remote;
            if (rv != prev)
            {
                ghost_vertices[ghost_count++] = rv;
                prev = rv;
            }
            boundary[i].remote = ghost_count - 1; /* overwrite with ghost index */
        }

        uint32_t *shr = (uint32_t*)realloc(ghost_vertices, (size_t)ghost_count * sizeof(uint32_t));
        if (shr) ghost_vertices = shr;

        ghost_labels = (uint32_t*)malloc((size_t)ghost_count * sizeof(uint32_t));
        if (!ghost_labels) die_abort(comm, "OOM: ghost_labels");
        for (uint32_t i = 0; i < ghost_count; ++i)
            ghost_labels[i] = ghost_vertices[i];
    }

    double t_g1 = MPI_Wtime();
    printf("[rank %d] ghost_count=%u build_ghosts=%.3fs\n", rank, ghost_count, t_g1 - t_g0);

    /* ============================================================
       Exchange plan
       ============================================================ */
    double t_p0 = MPI_Wtime();

    ExchangePlan plan;
    memset(&plan, 0, sizeof(plan));
    if (ghost_count > 0)
        exchangeplan_build(&plan, ghost_vertices, ghost_count, n_global, comm, owner_of_vertex);

    double t_p1 = MPI_Wtime();
    printf("[rank %d] plan indegree=%d outdegree=%d send=%d recv=%d build_plan=%.3fs\n",
           rank, plan.indegree, plan.outdegree, plan.total_send_to, plan.total_need_from, t_p1 - t_p0);

    /* ============================================================
       Merge loop: exchange + hook + pointer-jump
       Persistent thread pool
       ============================================================ */
    int global_changed = (ghost_count > 0 && boundary_sz > 0 && rep_count > 0) ? 1 : 0;

    pthread_t *threads = NULL;
    MergeWorker *workers = NULL;
    pthread_barrier_t bar;
    volatile int phase = PH_IDLE;
    volatile int changed = 0;

    if (nthreads < 1) nthreads = 1;

    if (nthreads > 1)
    {
        if (pthread_barrier_init(&bar, NULL, (unsigned)nthreads + 1u) != 0)
            die_abort(comm, "pthread_barrier_init failed");

        threads = (pthread_t*)malloc((size_t)nthreads * sizeof(pthread_t));
        workers = (MergeWorker*)malloc((size_t)nthreads * sizeof(MergeWorker));
        if (!threads || !workers) die_abort(comm, "OOM: thread pool");

        for (int t = 0; t < nthreads; ++t)
        {
            workers[t] = (MergeWorker){
                .tid = t,
                .nthreads = nthreads,
                .rep_list = rep_list,
                .rep_count = rep_count,
                .comp_label = comp_label,
                .comp_min = comp_min,
                .comp_of = comp_of,
                .v_start = v_start,
                .v_end = v_end,
                .ghost_vertices = ghost_vertices,
                .ghost_count = ghost_count,
                .ghost_labels = ghost_labels,
                .phase = &phase,
                .changed = &changed,
                .bar = &bar
            };
            if (pthread_create(&threads[t], NULL, merge_worker_main, &workers[t]) != 0)
                die_abort(comm, "pthread_create failed");
        }
    }

    MPI_Barrier(comm);

    int iter = 0;
    int steps_since_exchange = exchange_interval; /* exchange first iter */

    while (global_changed)
    {
        double it0 = MPI_Wtime();

        /* exchange interval */
        double t_pack0 = MPI_Wtime();
        double t_ex0 = t_pack0;

        if (steps_since_exchange >= exchange_interval)
        {
            /* pack outgoing labels */
            for (int i = 0; i < plan.total_send_to; ++i)
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
            double t_pack1 = MPI_Wtime();

            exchangeplan_exchange(&plan, ghost_labels, comm);
            double t_ex1 = MPI_Wtime();

            t_pack0 = t_pack0;
            t_ex0   = t_pack1; /* for printing; we’ll store actual below */
            (void)t_ex0;

            steps_since_exchange = 0;

            /* we’ll keep real times in locals for printing */
            double pack_dt = t_pack1 - it0;
            double exch_dt = t_ex1 - t_pack1;
            (void)pack_dt;
            (void)exch_dt;
        }

        double t_pack1 = MPI_Wtime();

        /* PH_INIT in pool (or serial) */
        double t_init0 = MPI_Wtime();
        if (nthreads > 1)
        {
            phase = PH_INIT;
            pthread_barrier_wait(&bar);
            pthread_barrier_wait(&bar);
        }
        else
        {
            for (uint32_t i = 0; i < rep_count; ++i)
            {
                uint32_t r = rep_list[i];
                comp_min[r] = comp_label[r];
            }
        }
        double t_init1 = MPI_Wtime();

        /* SERIAL hook */
        double t_hook0 = MPI_Wtime();
        for (uint64_t e = 0; e < boundary_sz; ++e)
        {
            uint32_t rep  = boundary[e].rep;
            uint32_t gidx = boundary[e].remote;
            uint32_t lbl  = ghost_labels[gidx];
            if (lbl < comp_min[rep]) comp_min[rep] = lbl;
        }
        double t_hook1 = MPI_Wtime();

        /* PH_UPD in pool (or serial) */
        double t_upd0 = MPI_Wtime();
        changed = 0;

        if (nthreads > 1)
        {
            phase = PH_UPD;
            pthread_barrier_wait(&bar);
            pthread_barrier_wait(&bar);
        }
        else
        {
            int local_any = 0;
            const int MAX_JUMPS = 4;

            for (uint32_t i = 0; i < rep_count; ++i)
            {
                uint32_t r = rep_list[i];
                uint32_t x = comp_min[r];

                for (int t = 0; t < MAX_JUMPS; ++t)
                {
                    int missing = 0;
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

                if (x < comp_min[r]) comp_min[r] = x;

                if (comp_min[r] < comp_label[r])
                {
                    comp_label[r] = comp_min[r];
                    local_any = 1;
                }
            }
            if (local_any) changed = 1;
        }
        double t_upd1 = MPI_Wtime();

        /* global changed */
        double t_allr0 = MPI_Wtime();
        MPI_Allreduce((void*)&changed, &global_changed, 1, MPI_INT, MPI_LOR, comm);
        double t_allr1 = MPI_Wtime();

        iter++;
        steps_since_exchange++;

        double it1 = MPI_Wtime();

        if (iter <= 3 || (iter % 10 == 0))
        {
            /* pack/exch timings: we only exchange on interval; report “pack/exch” as time since it0 to t_pack1 */
            printf("[rank %d] iter=%d pack+exch=%.3fs init=%.3fs hook=%.3fs upd=%.3fs allr=%.3fs total=%.3fs changed=%d\n",
                   rank, iter,
                   (t_pack1 - it0),
                   (t_init1 - t_init0),
                   (t_hook1 - t_hook0),
                   (t_upd1  - t_upd0),
                   (t_allr1 - t_allr0),
                   (it1 - it0),
                   global_changed);
        }
    }

    if (rank == 0)
        printf("CC merge converged in %d rounds\n", iter);

    /* stop pool */
    if (nthreads > 1)
    {
        phase = PH_STOP;
        pthread_barrier_wait(&bar);
        pthread_barrier_wait(&bar);

        for (int t = 0; t < nthreads; ++t)
            pthread_join(threads[t], NULL);

        pthread_barrier_destroy(&bar);

        free(threads);
        free(workers);
    }

    /* ============================================================
       Exact CC count (no gather): count roots (label == itself) exactly once
       ============================================================ */
    uint64_t local_roots = 0;
    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint32_t rep = comp_of[li];
        if (comp_label[rep] == v_start + li)
            local_roots++;
    }

    uint64_t global_roots = 0;
    MPI_Reduce(&local_roots, &global_roots, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

    if (rank == 0)
        printf("Number of connected components: %llu\n", (unsigned long long)global_roots);

    double t_total1 = MPI_Wtime();
    if (rank == 0)
        printf("[rank 0] total_cc_time=%.3fs\n", t_total1 - t_total0);

    /* ============================================================
       Cleanup
       ============================================================ */
    exchangeplan_free(&plan);

    free(boundary);
    free(ghost_vertices);
    free(ghost_labels);

    free(rep_list);
    free(is_rep);

    free(comp_of);
    free(comp_label);
    free(comp_min);
}

/* Keep this helper for the “gather labels” optional path in your benchmark */
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
    for (uint32_t v = 0; v < n_global; ++v)
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
