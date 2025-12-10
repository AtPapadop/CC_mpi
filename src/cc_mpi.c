#define _POSIX_C_SOURCE 200112L

#include "cc_mpi.h"
#include <mpi.h>
#include <omp.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MPI_INT32_T
#define MPI_INT32_T MPI_INT
#endif

#ifndef MPI_INT64_T
#define MPI_INT64_T MPI_LONG_LONG
#endif

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif

#define LABEL_TAG 1234
#define ALIGN_BYTES 64

/* Simple dynamic array for uint32_t */
typedef struct
{
    uint32_t *data;
    int size;
    int capacity;
} Int32Vec;

static void vec_init(Int32Vec *v)
{
    v->data = NULL;
    v->size = 0;
    v->capacity = 0;
}

static void vec_reserve(Int32Vec *v, int new_cap)
{
    if (new_cap <= v->capacity)
        return;

    uint32_t *new_data = (uint32_t *)realloc(v->data, (size_t)new_cap * sizeof(uint32_t));
    if (!new_data)
    {
        fprintf(stderr, "vec_reserve: out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    v->data = new_data;
    v->capacity = new_cap;
}

static void vec_push(Int32Vec *v, uint32_t x)
{
    if (v->size == v->capacity)
    {
        int new_cap = (v->capacity > 0) ? 2 * v->capacity : 16;
        vec_reserve(v, new_cap);
    }
    v->data[v->size++] = x;
}

static void vec_free(Int32Vec *v)
{
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->capacity = 0;
}

/* Owner of vertex v under the same block partition. */
static inline int owner_of_vertex(uint32_t v,
                                  uint32_t n_global,
                                  int comm_size)
{
    if (comm_size <= 0)
        return 0;

    uint32_t comm_size_u = (uint32_t)comm_size;
    uint32_t base = (comm_size_u > 0) ? (n_global / comm_size_u) : 0;
    uint32_t rem  = (comm_size_u > 0) ? (n_global % comm_size_u) : 0;
    uint64_t threshold = (uint64_t)(base + 1u) * rem;

    if ((uint64_t)v < threshold)
    {
        uint32_t denom = base + 1u;
        return denom ? (int)(v / denom) : 0;
    }

    if (base == 0)
        return (int)rem;

    uint64_t shifted = (uint64_t)v - threshold;
    return (int)(rem + (uint32_t)(shifted / base));
}

/* 64-byte aligned malloc with MPI_Abort on error (for hot arrays). */
static void *xaligned_alloc_or_die(size_t nbytes)
{
    void *ptr = NULL;
    if (nbytes == 0)
        return NULL;

    int rc = posix_memalign(&ptr, ALIGN_BYTES, nbytes);
    if (rc != 0 || !ptr)
    {
        fprintf(stderr, "xaligned_alloc_or_die: posix_memalign failed for %zu bytes (rc=%d)\n",
                nbytes, rc);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return ptr;
}

/* Get current label (component representative) for global vertex v. */
static inline uint32_t get_label_global(uint32_t v,
                                        const uint32_t *restrict local_labels,
                                        const uint32_t *restrict ghost_labels,
                                        const uint32_t *restrict ghost_index,
                                        uint32_t v_start,
                                        uint32_t v_end)
{
    if (v >= v_start && v < v_end)
    {
        return local_labels[v - v_start];
    }

    uint32_t gi = ghost_index[v];
    if (gi != UINT32_MAX)
        return ghost_labels[gi];

    return v; /* fallback: treat as its own representative */
}

/* -------------------------------------------------------------------------- */
/*  Connected components: distributed hooking + shortcutting (union-find-ish) */
/* -------------------------------------------------------------------------- */

void compute_connected_components_mpi_advanced(const DistCSRGraph *restrict Gd,
                                               uint32_t *restrict labels_global,
                                               int chunk_size,
                                               int exchange_interval,
                                               MPI_Comm comm)
{
    int comm_size = 0, comm_rank = 0;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    const uint32_t n_global = Gd->n_global;
    const uint64_t *row_ptr = Gd->row_ptr; /* local row_ptr (size n_local+1) */
    const uint32_t *col_idx = Gd->col_idx; /* local col_idx (size m_local) */
    const uint32_t v_start  = Gd->v_start;
    const uint32_t v_end    = Gd->v_end;
    const uint32_t n_local  = Gd->n_local;

    if (n_global == 0)
        return;

    /* Effective exchange interval (how many local sweeps before halo exchange) */
    int eff_exchange = (exchange_interval > 0) ? exchange_interval : 1;

    /* Local label arrays (double-buffered), 64-byte aligned. */
    uint32_t *local_labels     = NULL;
    uint32_t *local_labels_new = NULL;

    if (n_local > 0)
    {
        local_labels     = (uint32_t *)xaligned_alloc_or_die((size_t)n_local * sizeof(uint32_t));
        local_labels_new = (uint32_t *)xaligned_alloc_or_die((size_t)n_local * sizeof(uint32_t));

        /* Initialize labels to vertex IDs (global) */
#pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i < n_local; ++i)
        {
            uint32_t v = v_start + i;
            local_labels[i]     = v;
            local_labels_new[i] = v;
        }
    }

    /* Ghost mapping: map global vertex -> ghost index (0..ghost_count-1).
     * We use an array of size n_global for O(1) lookup.
     */
    uint32_t *ghost_index = (uint32_t *)xaligned_alloc_or_die((size_t)n_global * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_global; ++i)
    {
        ghost_index[i] = UINT32_MAX;
    }

    /* need_from[p]: which global vertices (owned by rank p) we need labels for */
    Int32Vec *need_from = (Int32Vec *)malloc((size_t)comm_size * sizeof(Int32Vec));
    if (!need_from)
    {
        fprintf(stderr, "[rank %d] Failed to allocate need_from\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    for (int p = 0; p < comm_size; ++p)
        vec_init(&need_from[p]);

    /* Mark array to avoid duplicates when collecting ghosts. */
    unsigned char *mark = (unsigned char *)calloc((size_t)n_global, sizeof(unsigned char));
    if (!mark)
    {
        fprintf(stderr, "[rank %d] Failed to allocate mark array\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Discover ghost vertices from local adjacency. */
    for (uint32_t li = 0; li < n_local; ++li)
    {
        uint64_t row_begin = row_ptr[li];
        uint64_t row_end   = row_ptr[li + 1];

        for (uint64_t j = row_begin; j < row_end; ++j)
        {
            uint32_t v = col_idx[j];
            if (v >= n_global)
                continue;

            if (v >= v_start && v < v_end)
            {
                /* Local neighbor */
                continue;
            }

            int owner = owner_of_vertex(v, n_global, comm_size);
            if (owner == comm_rank)
                continue;

            if (!mark[v])
            {
                mark[v] = 1;
                vec_push(&need_from[owner], v);
            }
        }
    }

    free(mark);
    mark = NULL;

    /* Assign ghost indices and allocate ghost label array. */
    uint32_t ghost_count = 0;
    for (int p = 0; p < comm_size; ++p)
    {
        Int32Vec *vec = &need_from[p];
        for (int i = 0; i < vec->size; ++i)
        {
            uint32_t v = vec->data[i];
            ghost_index[v] = ghost_count++;
        }
    }

    uint32_t *ghost_labels      = NULL;
    uint32_t *ghost_labels_prev = NULL;
    if (ghost_count > 0)
    {
        ghost_labels      = (uint32_t *)xaligned_alloc_or_die((size_t)ghost_count * sizeof(uint32_t));
        ghost_labels_prev = (uint32_t *)xaligned_alloc_or_die((size_t)ghost_count * sizeof(uint32_t));

        /* Initialize ghosts to their own vertex IDs. */
        for (int p = 0; p < comm_size; ++p)
        {
            Int32Vec *vec = &need_from[p];
            for (int i = 0; i < vec->size; ++i)
            {
                uint32_t v = vec->data[i];
                uint32_t idx = ghost_index[v];
                if (idx != UINT32_MAX)
                {
                    ghost_labels[idx]      = v;
                    ghost_labels_prev[idx] = v;
                }
            }
        }
    }

    /* Handshake: compute who we must send labels to (send_to). */
    int *need_from_counts = (int *)malloc((size_t)comm_size * sizeof(int));
    int *send_to_counts   = (int *)malloc((size_t)comm_size * sizeof(int));
    if (!need_from_counts || !send_to_counts)
    {
        fprintf(stderr, "[rank %d] Failed to allocate counts arrays\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for (int p = 0; p < comm_size; ++p)
    {
        need_from_counts[p] = need_from[p].size;
    }

    /* send_to_counts[p] = how many vertices rank p needs from us */
    MPI_Alltoall(need_from_counts, 1, MPI_INT,
                 send_to_counts,   1, MPI_INT,
                 comm);

    /* Flatten the need_from arrays for all-to-all. */
    int *need_from_displs = (int *)malloc((size_t)comm_size * sizeof(int));
    int *send_to_displs   = (int *)malloc((size_t)comm_size * sizeof(int));
    if (!need_from_displs || !send_to_displs)
    {
        fprintf(stderr, "[rank %d] Failed to allocate displs arrays\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    int total_need_from = 0;
    for (int p = 0; p < comm_size; ++p)
    {
        need_from_displs[p] = total_need_from;
        total_need_from += need_from_counts[p];
    }

    int total_send_to = 0;
    for (int p = 0; p < comm_size; ++p)
    {
        send_to_displs[p] = total_send_to;
        total_send_to += send_to_counts[p];
    }

    uint32_t *need_from_flat = NULL;
    uint32_t *send_to_flat   = NULL;
    if (total_need_from > 0)
    {
        need_from_flat = (uint32_t *)malloc((size_t)total_need_from * sizeof(uint32_t));
        if (!need_from_flat)
        {
            fprintf(stderr, "[rank %d] Failed to allocate need_from_flat\n", comm_rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        for (int p = 0; p < comm_size; ++p)
        {
            if (need_from_counts[p] > 0)
            {
                memcpy(need_from_flat + need_from_displs[p],
                       need_from[p].data,
                       (size_t)need_from_counts[p] * sizeof(uint32_t));
            }
        }
    }

    if (total_send_to > 0)
    {
        send_to_flat = (uint32_t *)malloc((size_t)total_send_to * sizeof(uint32_t));
        if (!send_to_flat)
        {
            fprintf(stderr, "[rank %d] Failed to allocate send_to_flat\n", comm_rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    /* All-to-all exchange of vertex IDs we must send labels for. */
    MPI_Alltoallv(need_from_flat, need_from_counts, need_from_displs, MPI_UINT32_T,
                  send_to_flat,  send_to_counts,   send_to_displs,   MPI_UINT32_T,
                  comm);

    /* Build send_to views that reference send_to_flat. */
    Int32Vec *send_to = (Int32Vec *)malloc((size_t)comm_size * sizeof(Int32Vec));
    if (!send_to)
    {
        fprintf(stderr, "[rank %d] Failed to allocate send_to\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for (int p = 0; p < comm_size; ++p)
    {
        vec_init(&send_to[p]);
        int cnt = send_to_counts[p];
        if (cnt > 0)
        {
            send_to[p].data     = send_to_flat + send_to_displs[p];
            send_to[p].size     = cnt;
            send_to[p].capacity = cnt; /* view; do not free individually */
        }
    }

    /* ===================== NEW: MPI distributed-graph communicator ==================== */

    /* Build list of incoming and outgoing neighbors for dist graph. */
    int indegree  = 0;
    int outdegree = 0;
    for (int p = 0; p < comm_size; ++p)
    {
        if (need_from_counts[p] > 0) ++indegree;
        if (send_to_counts[p]   > 0) ++outdegree;
    }

    int *sources      = (indegree  > 0) ? (int *)malloc((size_t)indegree  * sizeof(int)) : NULL;
    int *destinations = (outdegree > 0) ? (int *)malloc((size_t)outdegree * sizeof(int)) : NULL;

    if ((indegree  > 0 && !sources) ||
        (outdegree > 0 && !destinations))
    {
        fprintf(stderr, "[rank %d] Failed to allocate sources/destinations\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    int idx_in = 0;
    int idx_out = 0;
    for (int p = 0; p < comm_size; ++p)
    {
        if (need_from_counts[p] > 0) sources[idx_in++]   = p;
        if (send_to_counts[p]   > 0) destinations[idx_out++] = p;
    }

    MPI_Comm comm_graph = MPI_COMM_NULL;
    MPI_Dist_graph_create_adjacent(
        comm,
        indegree,  sources,      MPI_UNWEIGHTED,
        outdegree, destinations, MPI_UNWEIGHTED,
        MPI_INFO_NULL, 0, &comm_graph);

    /* Neigh all-to-allv send/recv layout (indices are neighbor indices, not ranks). */
    int *sendcounts = (outdegree > 0) ? (int *)malloc((size_t)outdegree * sizeof(int)) : NULL;
    int *sdispls    = (outdegree > 0) ? (int *)malloc((size_t)outdegree * sizeof(int)) : NULL;
    int *recvcounts = (indegree  > 0) ? (int *)malloc((size_t)indegree  * sizeof(int)) : NULL;
    int *rdispls    = (indegree  > 0) ? (int *)malloc((size_t)indegree  * sizeof(int)) : NULL;

    if ((outdegree > 0 && (!sendcounts || !sdispls)) ||
        (indegree  > 0 && (!recvcounts || !rdispls)))
    {
        fprintf(stderr, "[rank %d] Failed to allocate neighbor counts/displs\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (outdegree > 0)
    {
        for (int k = 0; k < outdegree; ++k)
        {
            int p = destinations[k];
            sendcounts[k] = send_to_counts[p];
            sdispls[k]    = send_to_displs[p];
        }
    }
    if (indegree > 0)
    {
        for (int k = 0; k < indegree; ++k)
        {
            int p = sources[k];
            recvcounts[k] = need_from_counts[p];
            rdispls[k]    = need_from_displs[p];
        }
    }

    /* Flat label buffers that match send_to_flat/need_from_flat layout. */
    uint32_t *send_labels_flat = (total_send_to   > 0)
                                 ? (uint32_t *)xaligned_alloc_or_die((size_t)total_send_to   * sizeof(uint32_t))
                                 : NULL;
    uint32_t *recv_labels_flat = (total_need_from > 0)
                                 ? (uint32_t *)xaligned_alloc_or_die((size_t)total_need_from * sizeof(uint32_t))
                                 : NULL;

    /* ===================== END NEW COMM SETUP ======================================== */

    /* OpenMP scheduling */
    const int DEFAULT_CHUNK_SIZE = 4096;
    const int chunking_enabled   = (chunk_size != 1);
    const int effective_chunk    = (chunk_size > 0) ? chunk_size : DEFAULT_CHUNK_SIZE;

    omp_set_schedule(chunking_enabled ? omp_sched_dynamic : omp_sched_static,
                     chunking_enabled ? effective_chunk : 0);

    bool global_changed = true;
    int  iterations     = 0;

    while (global_changed)
    {
        bool local_changed_epoch = false;

        /* ---- Inner loop: multiple local iterations with stale ghosts ---- */
        for (int step = 0; step < eff_exchange; ++step)
        {
            bool local_changed = false;

            if (n_local > 0)
            {
#pragma omp parallel for schedule(runtime) reduction(|| : local_changed)
                for (uint32_t li = 0; li < n_local; ++li)
                {
                    uint32_t old_label = local_labels[li];
                    uint32_t new_label = old_label;

                    uint64_t row_begin = row_ptr[li];
                    uint64_t row_end   = row_ptr[li + 1];

                    /* Hooking: take minimum label over self and neighbors. */
                    for (uint64_t j = row_begin; j < row_end; ++j)
                    {
                        uint32_t v = col_idx[j];
                        if (v >= n_global)
                            continue;

                        uint32_t neighbor_label;
                        if (v >= v_start && v < v_end)
                        {
                            /* Local neighbor */
                            neighbor_label = local_labels[v - v_start];
                        }
                        else
                        {
                            uint32_t gidx = ghost_index[v];
                            if (gidx != UINT32_MAX)
                                neighbor_label = ghost_labels[gidx];
                            else
                                neighbor_label = v; /* safety fallback */
                        }

                        if (neighbor_label < new_label)
                            new_label = neighbor_label;
                    }

                    /* Shortcutting (one step of pointer jumping):
                     * new_label := label[new_label].
                     */
                    if (new_label != old_label)
                    {
                        uint32_t root = get_label_global(new_label,
                                                         local_labels, ghost_labels, ghost_index,
                                                         v_start, v_end);
                        if (root < new_label)
                            new_label = root;
                    }

                    local_labels_new[li] = new_label;
                    if (new_label != old_label)
                        local_changed = true;
                }

                /* Swap local label buffers */
                uint32_t *tmp = local_labels;
                local_labels = local_labels_new;
                local_labels_new = tmp;
            }

            if (local_changed)
                local_changed_epoch = true;

            ++iterations;
        }

        /* ---- Halo exchange after the epoch, via MPI_Neighbor_alltoallv ---- */

        bool ghosts_changed = false;
        if (ghost_count > 0)
        {
            memcpy(ghost_labels_prev, ghost_labels, (size_t)ghost_count * sizeof(uint32_t));
        }

        /* Pack outgoing labels for all boundary vertices into send_labels_flat. */
        if (total_send_to > 0)
        {
            for (int p = 0; p < comm_size; ++p)
            {
                int cnt = send_to_counts[p];
                if (cnt <= 0)
                    continue;

                int base = send_to_displs[p];
                uint32_t *verts = send_to[p].data;

                for (int i = 0; i < cnt; ++i)
                {
                    uint32_t v = verts[i];
                    uint32_t lbl = v; /* fallback */

                    if (v >= v_start && v < v_end)
                    {
                        uint32_t li = v - v_start;
                        if (li < n_local)
                            lbl = local_labels[li];
                    }

                    send_labels_flat[base + i] = lbl;
                }
            }
        }

        if (comm_graph != MPI_COMM_NULL &&
            (indegree > 0 || outdegree > 0) &&
            (total_send_to > 0 || total_need_from > 0))
        {
            MPI_Neighbor_alltoallv(send_labels_flat, sendcounts, sdispls, MPI_UINT32_T,
                                   recv_labels_flat, recvcounts, rdispls, MPI_UINT32_T,
                                   comm_graph);
        }

        /* Update ghost labels from received data. */
        if (ghost_count > 0 && total_need_from > 0)
        {
            for (int p = 0; p < comm_size; ++p)
            {
                int cnt = need_from_counts[p];
                if (cnt <= 0)
                    continue;

                int base = need_from_displs[p];
                uint32_t *verts = need_from_flat + base;

                for (int i = 0; i < cnt; ++i)
                {
                    uint32_t v = verts[i];
                    uint32_t idx = ghost_index[v];
                    if (idx != UINT32_MAX)
                        ghost_labels[idx] = recv_labels_flat[base + i];
                }
            }

            /* Check if any ghost label changed due to this exchange. */
            for (uint32_t i = 0; i < ghost_count; ++i)
            {
                if (ghost_labels[i] != ghost_labels_prev[i])
                {
                    ghosts_changed = true;
                    break;
                }
            }
        }

        bool local_or_ghost_changed = local_changed_epoch || ghosts_changed;

        /* Global convergence: stop only when no rank saw any local or ghost changes. */
        MPI_Allreduce(&local_or_ghost_changed, &global_changed, 1,
                      MPI_C_BOOL, MPI_LOR, comm);
    }

    if (comm_rank == 0)
    {
        fprintf(stderr, "CC MPI+OMP (fast, graph-topo, interval=%d) converged in %d iterations\n",
                eff_exchange, iterations);
    }

    /* ---- Gather global membership vector on all ranks ---- */
    uint32_t base = (comm_size > 0) ? (n_global / (uint32_t)comm_size) : 0;
    uint32_t rem  = (comm_size > 0) ? (n_global % (uint32_t)comm_size) : 0;

    int *all_counts = (int *)malloc((size_t)comm_size * sizeof(int));
    int *all_displs = (int *)malloc((size_t)comm_size * sizeof(int));
    if (!all_counts || !all_displs)
    {
        fprintf(stderr, "[rank %d] Failed to allocate gather counts/displs\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    int offset = 0;
    for (int r = 0; r < comm_size; ++r)
    {
        uint32_t ln = ((uint32_t)r < rem) ? (base + 1u) : base;
        all_counts[r] = (int)ln;
        all_displs[r] = offset;
        offset += (int)ln;
    }

    MPI_Allgatherv(local_labels, (int)n_local, MPI_UINT32_T,
                   labels_global, all_counts, all_displs, MPI_UINT32_T,
                   comm);

    /* ---- Cleanup ---- */
    free(all_counts);
    free(all_displs);

    /* Free communicator-related stuff */
    if (comm_graph != MPI_COMM_NULL)
        MPI_Comm_free(&comm_graph);

    free(sources);
    free(destinations);
    free(sendcounts);
    free(sdispls);
    free(recvcounts);
    free(rdispls);
    if (send_labels_flat)
        free(send_labels_flat);
    if (recv_labels_flat)
        free(recv_labels_flat);

    for (int p = 0; p < comm_size; ++p)
        vec_free(&need_from[p]);
    free(need_from);
    free(send_to);

    free(need_from_counts);
    free(send_to_counts);
    free(need_from_displs);
    free(send_to_displs);

    free(need_from_flat);
    free(send_to_flat);

    if (ghost_labels)
        free(ghost_labels);
    if (ghost_labels_prev)
        free(ghost_labels_prev);
    if (ghost_index)
        free(ghost_index);

    if (local_labels)
        free(local_labels);
    if (local_labels_new)
        free(local_labels_new);
}

/* Same as your original helper; kept for completeness. */
uint32_t count_connected_components(const uint32_t *restrict labels_global, uint32_t n_global)
{
    if (n_global == 0 || labels_global == NULL)
        return 0;

    /* Use a bitmap to track unique component representatives */
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
