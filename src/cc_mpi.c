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
#include <inttypes.h>

#include "graph.h"
#include "cc.h"
#include "exchange_plan.h"
#include "vec_helpers.h"
#include "runtime_utils.h"
#include "boundary_sort.h"

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif

static int g_cc_mpi_threads = 0;

void cc_mpi_set_num_threads(int nthreads)
{
    g_cc_mpi_threads = (nthreads > 0) ? nthreads : 0;
}

static const uint32_t *g_part_bounds = NULL;
static int g_part_size = 0;
static int g_part_kind = 0;

static int owner_of_vertex(uint32_t v, uint32_t n_global, int comm_size)
{
    if (g_part_kind != 0 && g_part_bounds && g_part_size == comm_size && comm_size > 1)
    {
        int lo = 0;
        int hi = comm_size - 1;
        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            uint32_t a = g_part_bounds[mid];
            uint32_t b = g_part_bounds[mid + 1];
            if (v < a)
                hi = mid - 1;
            else if (v >= b)
                lo = mid + 1;
            else
                return mid;
        }
        return (v >= g_part_bounds[comm_size]) ? (comm_size - 1) : 0;
    }

    if (comm_size <= 1)
        return 0;

    uint32_t cs = (uint32_t)comm_size;
    uint32_t base = n_global / cs;
    uint32_t rem = n_global % cs;

    uint64_t split = (uint64_t)(base + 1u) * (uint64_t)rem;
    if ((uint64_t)v < split)
        return (int)(v / (base + 1u));

    if (base == 0)
        return (int)rem;
    return (int)(rem + (uint32_t)(((uint64_t)v - split) / base));
}

/* --------------------------
   Ghost map: uint32 -> uint32
   (open addressing, linear probing)
   Used to map global vertex id -> ghost index (gidx)
   -------------------------- */

typedef struct
{
    uint32_t *keys; /* global vertex ids */
    uint32_t *vals; /* gidx */
    uint32_t cap;   /* power-of-two */
    uint32_t size;

    int has_max;
    uint32_t max_val;
} GhostMap;

static inline uint32_t hash_u32(uint32_t x)
{
    /* decent 32-bit mix */
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static uint32_t next_pow2_u32_u64(uint64_t x)
{
    if (x <= 2)
        return 2;
    uint64_t p = 1;
    while (p < x)
        p <<= 1;
    if (p > (uint64_t)UINT32_MAX)
        return 0;
    return (uint32_t)p;
}

static void ghostmap_free(GhostMap *M)
{
    if (!M)
        return;
    free(M->keys);
    free(M->vals);
    memset(M, 0, sizeof(*M));
}

static void ghostmap_init(GhostMap *M, uint32_t cap_pow2, MPI_Comm comm)
{
    memset(M, 0, sizeof(*M));
    if (cap_pow2 < 1024)
        cap_pow2 = 1024;
    /* must be power of two */
    if ((cap_pow2 & (cap_pow2 - 1u)) != 0)
        mpi_die_abort(comm, "ghostmap_init: cap not power-of-two");

    M->cap = cap_pow2;
    M->keys = (uint32_t *)malloc((size_t)M->cap * sizeof(uint32_t));
    M->vals = (uint32_t *)malloc((size_t)M->cap * sizeof(uint32_t));
    if (!M->keys || !M->vals)
        mpi_die_abort(comm, "ghostmap_init: OOM");

    /* EMPTY = UINT32_MAX */
    for (uint32_t i = 0; i < M->cap; ++i)
        M->keys[i] = UINT32_MAX;
}

static void ghostmap_rehash(GhostMap *M, uint32_t newcap, MPI_Comm comm)
{
    GhostMap N;
    ghostmap_init(&N, newcap, comm);

    /* reinsert */
    for (uint32_t i = 0; i < M->cap; ++i)
    {
        uint32_t k = M->keys[i];
        if (k == UINT32_MAX)
            continue;
        uint32_t v = M->vals[i];

        uint32_t mask = N.cap - 1u;
        uint32_t idx = hash_u32(k) & mask;
        while (N.keys[idx] != UINT32_MAX)
            idx = (idx + 1u) & mask;
        N.keys[idx] = k;
        N.vals[idx] = v;
        N.size++;
    }

    if (M->has_max)
    {
        N.has_max = 1;
        N.max_val = M->max_val;
    }

    ghostmap_free(M);
    *M = N;
}

static inline int ghostmap_get(const GhostMap *M, uint32_t key, uint32_t *out_val)
{
    if (key == UINT32_MAX)
    {
        if (M->has_max)
        {
            *out_val = M->max_val;
            return 1;
        }
        return 0;
    }

    uint32_t mask = M->cap - 1u;
    uint32_t idx = hash_u32(key) & mask;

    while (1)
    {
        uint32_t k = M->keys[idx];
        if (k == UINT32_MAX)
            return 0;
        if (k == key)
        {
            *out_val = M->vals[idx];
            return 1;
        }
        idx = (idx + 1u) & mask;
    }
}

static inline void ghostmap_put(GhostMap *M, uint32_t key, uint32_t val, MPI_Comm comm)
{
    if (key == UINT32_MAX)
    {
        M->has_max = 1;
        M->max_val = val;
        return;
    }

    /* grow at ~0.70 load */
    if ((uint64_t)(M->size + 1u) * 10ull >= (uint64_t)M->cap * 7ull)
    {
        uint32_t newcap = (M->cap <= (UINT32_MAX / 2u)) ? (M->cap << 1u) : 0;
        if (!newcap)
            mpi_die_abort(comm, "ghostmap_put: cap overflow");
        ghostmap_rehash(M, newcap, comm);
    }

    uint32_t mask = M->cap - 1u;
    uint32_t idx = hash_u32(key) & mask;
    while (M->keys[idx] != UINT32_MAX)
        idx = (idx + 1u) & mask;

    M->keys[idx] = key;
    M->vals[idx] = val;
    M->size++;
}

static inline void ensure_u32_array_cap(uint32_t **arr, uint32_t *cap_io, uint32_t need, MPI_Comm comm)
{
    uint32_t cap = *cap_io;
    if (cap >= need)
        return;

    uint32_t newcap = cap ? cap : 1024u;
    while (newcap < need)
    {
        if (newcap > (UINT32_MAX / 2u))
            mpi_die_abort(comm, "ensure_u32_array_cap: cap overflow");
        newcap <<= 1u;
    }

    uint32_t *nb = (uint32_t *)realloc(*arr, (size_t)newcap * sizeof(uint32_t));
    if (!nb)
        mpi_die_abort(comm, "ensure_u32_array_cap: OOM");
    *arr = nb;
    *cap_io = newcap;
}

static inline void ensure_boundary_cap(BoundaryEdge **arr, uint64_t *cap_io, uint64_t need, MPI_Comm comm)
{
    uint64_t cap = *cap_io;
    if (cap >= need)
        return;

    uint64_t newcap = cap ? cap : 1024ull;
    while (newcap < need)
        newcap <<= 1ull;

    if (newcap > (uint64_t)(SIZE_MAX / sizeof(BoundaryEdge)))
        mpi_die_abort(comm, "ensure_boundary_cap: size overflow");

    BoundaryEdge *nb = (BoundaryEdge *)realloc(*arr, (size_t)newcap * sizeof(BoundaryEdge));
    if (!nb)
        mpi_die_abort(comm, "ensure_boundary_cap: OOM");
    *arr = nb;
    *cap_io = newcap;
}

/* Return label(x) if known locally or as a ghost; otherwise missing=1. */
static inline uint32_t get_vertex_label_global(uint32_t x,
                                               const uint32_t *comp_label,
                                               const uint32_t *comp_of,
                                               uint32_t v_start, uint32_t v_end,
                                               const GhostMap *gmap,
                                               const uint32_t *ghost_labels,
                                               int *missing)
{
    if (x >= v_start && x < v_end)
    {
        uint32_t li = x - v_start;
        uint32_t rep = comp_of[li];
        if (missing)
            *missing = 0;
        return comp_label[rep];
    }

    uint32_t gidx = 0;
    if (gmap && ghost_labels && ghostmap_get(gmap, x, &gidx))
    {
        if (missing)
            *missing = 0;
        return ghost_labels[gidx];
    }

    if (missing)
        *missing = 1;
    return x;
}

enum
{
    PH_IDLE = 0,
    PH_INIT = 1,
    PH_UPD = 2,
    PH_STOP = 3
};

typedef struct
{
    int tid;
    int nthreads;

    const uint32_t *rep_list;
    uint32_t rep_count;

    uint32_t *comp_label;
    uint32_t *comp_min;

    const uint32_t *comp_of;
    uint32_t v_start, v_end;

    const GhostMap *gmap;
    const uint32_t *ghost_labels;

    volatile int *phase;
    int *changed;
    pthread_barrier_t *bar;
} MergeWorker;

static void *merge_worker_main(void *arg)
{
    MergeWorker *W = (MergeWorker *)arg;

    while (1)
    {
        pthread_barrier_wait(W->bar);

        int ph = *W->phase;
        if (ph == PH_STOP)
            break;

        uint32_t reps = W->rep_count;
        uint32_t start = (uint32_t)(((uint64_t)reps * (uint64_t)W->tid) / (uint64_t)W->nthreads);
        uint32_t end = (uint32_t)(((uint64_t)reps * (uint64_t)(W->tid + 1)) / (uint64_t)W->nthreads);

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
                    int miss = 0;
                    uint32_t y = get_vertex_label_global(x,
                                                         W->comp_label, W->comp_of,
                                                         W->v_start, W->v_end,
                                                         W->gmap, W->ghost_labels,
                                                         &miss);
                    if (miss)
                        break;
                    if (y >= x)
                        break;
                    x = y;
                }

                if (x < W->comp_min[r])
                    W->comp_min[r] = x;

                if (W->comp_min[r] < W->comp_label[r])
                {
                    W->comp_label[r] = W->comp_min[r];
                    local_any = 1;
                }
            }

            if (local_any)
                __atomic_store_n(W->changed, 1, __ATOMIC_RELAXED);
        }

        pthread_barrier_wait(W->bar);
    }

    pthread_barrier_wait(W->bar);
    return NULL;
}

void compute_connected_components_mpi_advanced(const DistCSRGraph *restrict Gd,
                                               uint32_t *restrict labels_global,
                                               int chunk_size,
                                               int exchange_interval,
                                               MPI_Comm comm)
{
    (void)labels_global;
    if (exchange_interval < 1)
        exchange_interval = 1;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    g_part_kind = Gd->part_kind;
    g_part_size = Gd->part_size;
    g_part_bounds = Gd->part_bounds;

    const uint32_t n_global = Gd->n_global;
    const uint32_t v_start = Gd->v_start;
    const uint32_t v_end = Gd->v_end;
    const uint32_t n_local = Gd->n_local;

    const uint64_t *row_ptr = Gd->row_ptr;
    const uint32_t *col_idx = Gd->col_idx;

    if (n_global == 0 || n_local == 0)
        return;

    double t_total0 = MPI_Wtime();

    uint64_t *lrow = (uint64_t *)calloc((size_t)n_local + 1, sizeof(uint64_t));
    if (!lrow)
        mpi_die_abort(comm, "OOM: lrow");

    uint64_t boundary_raw = 0;

    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint64_t begin = row_ptr[li];
        uint64_t end = row_ptr[li + 1];

        uint64_t cnt_local = 0;
        for (uint64_t j = begin; j < end; ++j)
        {
            uint32_t v = col_idx[j];
            if (v >= v_start && v < v_end)
                cnt_local++;
            else if (v < n_global)
                boundary_raw++;
        }
        lrow[li + 1] = lrow[li] + cnt_local;
    }

    uint64_t m_local_local = lrow[n_local];

    uint32_t *lcol = NULL;
    if (m_local_local > 0)
    {
        lcol = (uint32_t *)malloc((size_t)m_local_local * sizeof(uint32_t));
        if (!lcol)
            mpi_die_abort(comm, "OOM: lcol");
    }

    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint64_t write = lrow[li];
        uint64_t begin = row_ptr[li];
        uint64_t end = row_ptr[li + 1];

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

    uint32_t *comp_of = (uint32_t *)malloc((size_t)n_local * sizeof(uint32_t));
    if (!comp_of)
        mpi_die_abort(comm, "OOM: comp_of");

    int nthreads = (g_cc_mpi_threads > 0) ? g_cc_mpi_threads : runtime_default_threads();
    if (nthreads < 1)
        nthreads = 1;

    double t_cc0 = MPI_Wtime();
    compute_connected_components_pthreads(&Glocal, comp_of, nthreads, chunk_size);
    double t_cc1 = MPI_Wtime();

    printf("[rank %d] Local CC done: n_local=%u m_local=%llu threads=%d time=%.3fs\n",
           rank, n_local, (unsigned long long)m_local_local, nthreads, t_cc1 - t_cc0);

    free(lrow);
    free(lcol);
    Glocal.row_ptr = NULL;
    Glocal.col_idx = NULL;

    double t_rep0 = MPI_Wtime();

    uint8_t *is_rep = (uint8_t *)calloc((size_t)n_local, 1);
    if (!is_rep)
        mpi_die_abort(comm, "OOM: is_rep");

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

    uint32_t *rep_list = (uint32_t *)malloc((size_t)rep_count * sizeof(uint32_t));
    if (!rep_list)
        mpi_die_abort(comm, "OOM: rep_list");

    uint32_t wrep = 0;
    for (uint32_t r = 0; r < n_local; ++r)
        if (is_rep[r])
            rep_list[wrep++] = r;

    double t_rep1 = MPI_Wtime();
    if (rank == 0)
        printf("[rank 0] reps=%u build_reps=%.3fs\n", rep_count, t_rep1 - t_rep0);

    uint32_t *comp_label = (uint32_t *)malloc((size_t)n_local * sizeof(uint32_t));
    uint32_t *comp_min = (uint32_t *)malloc((size_t)n_local * sizeof(uint32_t));
    if (!comp_label || !comp_min)
        mpi_die_abort(comm, "OOM: comp_label/comp_min");

    for (uint32_t i = 0; i < n_local; ++i)
        comp_label[i] = v_start + i;

    /* -----------------------------------------------------------
       Build contracted boundary edges WITHOUT boundary_raw array
       - ghost map: global vertex -> gidx
       - last_seen[gidx] stores (rep+1) last rep that inserted it,
         so duplicates for the same rep are skipped.
       ----------------------------------------------------------- */
    double t_b0 = MPI_Wtime();

    BoundaryEdge *boundary = NULL;
    uint64_t boundary_sz = 0;
    uint64_t boundary_cap = 0;

    /* heuristic initial capacity ~ boundary_raw/12 */
    {
        uint64_t est = boundary_raw / 12ull + 1024ull;
        if (est < 1024ull)
            est = 1024ull;
        boundary_cap = est;
        if (boundary_cap > (uint64_t)(SIZE_MAX / sizeof(BoundaryEdge)))
            boundary_cap = (uint64_t)(SIZE_MAX / sizeof(BoundaryEdge));
        boundary = (BoundaryEdge *)malloc((size_t)boundary_cap * sizeof(BoundaryEdge));
        if (!boundary)
            mpi_die_abort(comm, "OOM: boundary (contracted)");
    }

    uint32_t *ghost_vertices = NULL;
    uint32_t *ghost_labels = NULL;
    uint32_t *ghost_last_seen = NULL;
    uint32_t ghost_cap = 0;
    uint32_t ghost_count = 0;

    GhostMap gmap;
    {
        /* heuristic initial map cap based on boundary_raw/16 */
        uint64_t gest = boundary_raw / 16ull + 4096ull;
        if (gest < 1024ull)
            gest = 1024ull;
        uint32_t cap = next_pow2_u32_u64(gest * 2ull);
        if (!cap)
            mpi_die_abort(comm, "ghost map cap too large");
        ghostmap_init(&gmap, cap, comm);
    }

    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint32_t rep = comp_of[li];

        uint64_t begin = row_ptr[li];
        uint64_t end = row_ptr[li + 1];

        for (uint64_t j = begin; j < end; ++j)
        {
            uint32_t v = col_idx[j];
            if (v >= v_start && v < v_end)
                continue;
            if (v >= n_global)
                continue;

            uint32_t gidx = 0;
            if (!ghostmap_get(&gmap, v, &gidx))
            {
                gidx = ghost_count;

                ensure_u32_array_cap(&ghost_vertices, &ghost_cap, ghost_count + 1u, comm);
                ensure_u32_array_cap(&ghost_labels, &ghost_cap, ghost_count + 1u, comm);
                ensure_u32_array_cap(&ghost_last_seen, &ghost_cap, ghost_count + 1u, comm);

                ghost_vertices[ghost_count] = v;
                ghost_labels[ghost_count] = v;
                ghost_last_seen[ghost_count] = 0;
                ghost_count++;

                ghostmap_put(&gmap, v, gidx, comm);
            }

            /* dedup per rep */
            uint32_t mark = rep + 1u; /* rep may be 0, so +1 avoids sentinel 0 */
            if (ghost_last_seen[gidx] == mark)
                continue;
            ghost_last_seen[gidx] = mark;

            /* append contracted boundary edge (rep, gidx) */
            if (boundary_sz == boundary_cap)
                ensure_boundary_cap(&boundary, &boundary_cap, boundary_sz + 1ull, comm);

            boundary[boundary_sz++] = (BoundaryEdge){rep, gidx};
        }
    }

    double t_b1 = MPI_Wtime();

    printf("[rank %d] boundary_contract raw=%llu contracted=%llu ghost_count=%u build=%.3fs\n",
           rank,
           (unsigned long long)boundary_raw,
           (unsigned long long)boundary_sz,
           ghost_count,
           t_b1 - t_b0);

    free(ghost_last_seen);
    ghost_last_seen = NULL;

    /* shrink boundary arrays (optional) */
    if (boundary_sz > 0)
    {
        BoundaryEdge *shr = (BoundaryEdge *)realloc(boundary, (size_t)boundary_sz * sizeof(BoundaryEdge));
        if (shr)
            boundary = shr;
    }
    if (ghost_count > 0)
    {
        uint32_t *shr1 = (uint32_t *)realloc(ghost_vertices, (size_t)ghost_count * sizeof(uint32_t));
        if (shr1)
            ghost_vertices = shr1;
        uint32_t *shr2 = (uint32_t *)realloc(ghost_labels, (size_t)ghost_count * sizeof(uint32_t));
        if (shr2)
            ghost_labels = shr2;
    }

    double t_p0 = MPI_Wtime();

    ExchangePlan plan;
    memset(&plan, 0, sizeof(plan));
    if (ghost_count > 0)
        exchangeplan_build(&plan, ghost_vertices, ghost_count, n_global, comm, owner_of_vertex);

    double t_p1 = MPI_Wtime();
    printf("[rank %d] plan indegree=%d outdegree=%d send=%d recv=%d build_plan=%.3fs\n",
           rank, plan.indegree, plan.outdegree, plan.total_send_to, plan.total_need_from, t_p1 - t_p0);

    uint32_t *prev_sent = NULL;
    uint64_t *delta_sendbuf = NULL;
    uint64_t *delta_recvbuf = NULL;
    int delta_sendcap = 0;
    int delta_recvcap = 0;

    if (plan.total_send_to > 0)
    {
        prev_sent = (uint32_t *)malloc((size_t)plan.total_send_to * sizeof(uint32_t));
        if (!prev_sent)
            mpi_die_abort(comm, "OOM: prev_sent");
        for (int i = 0; i < plan.total_send_to; ++i)
            prev_sent[i] = UINT32_MAX;
    }

    pthread_t *threads = NULL;
    MergeWorker *workers = NULL;
    pthread_barrier_t bar;
    volatile int phase = PH_IDLE;

    int changed = 0;
    int pending_since_exchange = 0;

    if (nthreads < 1)
        nthreads = 1;

    if (nthreads > 1)
    {
        if (pthread_barrier_init(&bar, NULL, (unsigned)nthreads + 1u) != 0)
            mpi_die_abort(comm, "pthread_barrier_init failed");

        threads = (pthread_t *)malloc((size_t)nthreads * sizeof(pthread_t));
        workers = (MergeWorker *)malloc((size_t)nthreads * sizeof(MergeWorker));
        if (!threads || !workers)
            mpi_die_abort(comm, "OOM: thread pool");

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
                .gmap = &gmap,
                .ghost_labels = ghost_labels,
                .phase = &phase,
                .changed = &changed,
                .bar = &bar};
            if (pthread_create(&threads[t], NULL, merge_worker_main, &workers[t]) != 0)
                mpi_die_abort(comm, "pthread_create failed");
        }
    }

    MPI_Barrier(comm);

    int iter = 0;
    int steps_since_exchange = exchange_interval;

    int global_pending = (ghost_count > 0 && boundary_sz > 0 && rep_count > 0) ? 1 : 0;
    pending_since_exchange = global_pending ? 1 : 0;

    while (global_pending)
    {
        double it0 = MPI_Wtime();
        int did_exchange = 0;

        if (steps_since_exchange >= exchange_interval)
        {
            if (plan.comm_graph != MPI_COMM_NULL)
            {
                exchangeplan_exchange_delta(&plan,
                                            comp_label,
                                            comp_of,
                                            v_start,
                                            v_end,
                                            prev_sent,
                                            ghost_labels,
                                            comm,
                                            &delta_sendbuf,
                                            &delta_sendcap,
                                            &delta_recvbuf,
                                            &delta_recvcap);
                did_exchange = 1;
                pending_since_exchange = 0;
            }

            steps_since_exchange = 0;
        }

        double t_pack1 = MPI_Wtime();

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

        double t_hook0 = MPI_Wtime();
        for (uint64_t e = 0; e < boundary_sz; ++e)
        {
            uint32_t rep = boundary[e].rep;
            uint32_t gidx = boundary[e].remote; /* remote holds gidx now */
            uint32_t lbl = ghost_labels[gidx];
            if (lbl < comp_min[rep])
                comp_min[rep] = lbl;
        }
        double t_hook1 = MPI_Wtime();

        double t_upd0 = MPI_Wtime();

        __atomic_store_n(&changed, 0, __ATOMIC_RELAXED);

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
                    int miss = 0;
                    uint32_t y = get_vertex_label_global(x,
                                                         comp_label, comp_of,
                                                         v_start, v_end,
                                                         &gmap, ghost_labels,
                                                         &miss);
                    if (miss)
                        break;
                    if (y >= x)
                        break;
                    x = y;
                }

                if (x < comp_min[r])
                    comp_min[r] = x;

                if (comp_min[r] < comp_label[r])
                {
                    comp_label[r] = comp_min[r];
                    local_any = 1;
                }
            }
            if (local_any)
                __atomic_store_n(&changed, 1, __ATOMIC_RELAXED);
        }

        int changed_snapshot = __atomic_load_n(&changed, __ATOMIC_RELAXED);

        if (changed_snapshot)
            pending_since_exchange = 1;

        double t_upd1 = MPI_Wtime();

        double t_allr0 = MPI_Wtime();

        int global_changed_any = 0;

        /* you can combine these two Allreduces later if you want */
        MPI_Allreduce(&pending_since_exchange, &global_pending, 1, MPI_INT, MPI_LOR, comm);
        MPI_Allreduce(&changed_snapshot, &global_changed_any, 1, MPI_INT, MPI_LOR, comm);

        double t_allr1 = MPI_Wtime();

        iter++;

        double it1 = MPI_Wtime();

        if (iter <= 3 || (iter % 10 == 0))
        {
            printf("[rank %d] iter=%d did_exch=%d pack+exch=%.3fs init=%.3fs hook=%.3fs upd=%.3fs allr=%.3fs total=%.3fs pending=%d changed=%d\n",
                   rank, iter,
                   did_exchange,
                   (t_pack1 - it0),
                   (t_init1 - t_init0),
                   (t_hook1 - t_hook0),
                   (t_upd1 - t_upd0),
                   (t_allr1 - t_allr0),
                   (it1 - it0),
                   global_pending,
                   changed_snapshot);
        }

        if (!global_pending)
            break;

        if (global_pending && !global_changed_any && steps_since_exchange < exchange_interval)
            steps_since_exchange = exchange_interval;

        steps_since_exchange++;
    }

    if (rank == 0)
        printf("CC merge converged in %d rounds\n", iter);

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

    free(prev_sent);
    free(delta_sendbuf);
    free(delta_recvbuf);
    exchangeplan_free(&plan);

    free(boundary);
    free(ghost_vertices);
    free(ghost_labels);
    ghostmap_free(&gmap);

    free(rep_list);
    free(is_rep);

    free(comp_of);
    free(comp_label);
    free(comp_min);
}

uint32_t count_connected_components(const uint32_t *restrict labels_global, uint32_t n_global)
{
    if (n_global == 0 || labels_global == NULL)
        return 0;

    unsigned char *comp_bitmap = (unsigned char *)calloc((size_t)n_global, sizeof(unsigned char));
    if (!comp_bitmap)
    {
        fprintf(stderr, "count_connected_components: Failed to allocate bitmap\n");
        return 0;
    }

    uint32_t comp_count = 0;
    for (uint32_t v = 0; v < n_global; ++v)
    {
        uint32_t rep = labels_global[v];
        if (rep >= n_global)
            continue;
        if (comp_bitmap[rep] == 0)
        {
            comp_bitmap[rep] = 1;
            comp_count++;
        }
    }

    free(comp_bitmap);
    return comp_count;
}
