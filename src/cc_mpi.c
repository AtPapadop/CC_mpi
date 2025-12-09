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

#define LABEL_TAG 1234

/* Simple dynamic array for int32_t */
typedef struct
{
    int32_t *data;
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

    int32_t *new_data = (int32_t *)realloc(v->data, (size_t)new_cap * sizeof(int32_t));
    if (!new_data)
    {
        fprintf(stderr, "vec_reserve: out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    v->data = new_data;
    v->capacity = new_cap;
}

static void vec_push(Int32Vec *v, int32_t x)
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
static inline int owner_of_vertex(int32_t v,
                                  int32_t n_global,
                                  int comm_size)
{
    (void)n_global; /* not needed for the math itself, kept for clarity */

    int32_t base = n_global / comm_size;
    int32_t rem = n_global % comm_size;

    int64_t threshold = (int64_t)(base + 1) * (int64_t)rem;

    if ((int64_t)v < threshold)
    {
        return v / (base + 1);
    }
    else
    {
        return rem + (v - (int32_t)threshold) / (base > 0 ? base : 1);
    }
}

void compute_connected_components_mpi_advanced(const DistCSRGraph *restrict Gd,
                                               int32_t *restrict labels_global,
                                               int chunk_size,
                                               int exchange_interval,
                                               MPI_Comm comm)
{
    int comm_size = 0, comm_rank = 0;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    const int32_t n_global = Gd->n_global;
    const int64_t *row_ptr = Gd->row_ptr; /* local row_ptr (size n_local+1) */
    const int32_t *col_idx = Gd->col_idx; /* local col_idx (size m_local) */
    const int32_t v_start = Gd->v_start;
    const int32_t v_end = Gd->v_end;
    const int32_t n_local = Gd->n_local;

    if (n_global == 0)
        return;

    /* Effective exchange interval */
    int eff_exchange = (exchange_interval > 0) ? exchange_interval : 1;

    /* Local label arrays (double-buffered) */
    int32_t *local_labels = NULL;
    int32_t *local_labels_new = NULL;

    if (n_local > 0)
    {
        local_labels = (int32_t *)malloc((size_t)n_local * sizeof(int32_t));
        local_labels_new = (int32_t *)malloc((size_t)n_local * sizeof(int32_t));
        if (!local_labels || !local_labels_new)
        {
            fprintf(stderr, "[rank %d] Failed to allocate local label arrays\n", comm_rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }

/* Initialize labels to vertex IDs (global) */
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < n_local; ++i)
        {
            int32_t v = v_start + i;
            local_labels[i] = v;
            local_labels_new[i] = v;
        }
    }

    /* Ghost mapping: map global vertex -> ghost index (0..ghost_count-1).
     * We use an array of size n_global for simplicity.
     */
    int32_t *ghost_index = (int32_t *)malloc((size_t)n_global * sizeof(int32_t));
    if (!ghost_index)
    {
        fprintf(stderr, "[rank %d] Failed to allocate ghost_index\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    for (int32_t i = 0; i < n_global; ++i)
    {
        ghost_index[i] = -1;
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
    for (int32_t li = 0; li < n_local; ++li)
    {
        int64_t row_begin = row_ptr[li];
        int64_t row_end = row_ptr[li + 1];

        for (int64_t j = row_begin; j < row_end; ++j)
        {
            int32_t v = col_idx[j];
            if (v < 0 || v >= n_global)
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
    int32_t ghost_count = 0;
    for (int p = 0; p < comm_size; ++p)
    {
        Int32Vec *vec = &need_from[p];
        for (int i = 0; i < vec->size; ++i)
        {
            int32_t v = vec->data[i];
            ghost_index[v] = ghost_count++;
        }
    }

    int32_t *ghost_labels = NULL;
    int32_t *ghost_labels_prev = NULL;
    if (ghost_count > 0)
    {
        ghost_labels = (int32_t *)malloc((size_t)ghost_count * sizeof(int32_t));
        ghost_labels_prev = (int32_t *)malloc((size_t)ghost_count * sizeof(int32_t));
        if (!ghost_labels || !ghost_labels_prev)
        {
            fprintf(stderr, "[rank %d] Failed to allocate ghost label arrays\n", comm_rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        /* Initialize ghosts to their own vertex IDs. */
        for (int p = 0; p < comm_size; ++p)
        {
            Int32Vec *vec = &need_from[p];
            for (int i = 0; i < vec->size; ++i)
            {
                int32_t v = vec->data[i];
                int32_t idx = ghost_index[v];
                ghost_labels[idx] = v;
            }
        }
    }

    /* Handshake: compute who we must send labels to (send_to). */

    int *need_from_counts = (int *)malloc((size_t)comm_size * sizeof(int));
    int *send_to_counts = (int *)malloc((size_t)comm_size * sizeof(int));
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
                 send_to_counts, 1, MPI_INT,
                 comm);

    /* Flatten the need_from arrays for all-to-all. */
    int *need_from_displs = (int *)malloc((size_t)comm_size * sizeof(int));
    int *send_to_displs = (int *)malloc((size_t)comm_size * sizeof(int));
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

    int32_t *need_from_flat = NULL;
    int32_t *send_to_flat = NULL;
    if (total_need_from > 0)
    {
        need_from_flat = (int32_t *)malloc((size_t)total_need_from * sizeof(int32_t));
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
                       (size_t)need_from_counts[p] * sizeof(int32_t));
            }
        }
    }

    if (total_send_to > 0)
    {
        send_to_flat = (int32_t *)malloc((size_t)total_send_to * sizeof(int32_t));
        if (!send_to_flat)
        {
            fprintf(stderr, "[rank %d] Failed to allocate send_to_flat\n", comm_rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    /* All-to-all exchange of vertex IDs we must send labels for. */
    MPI_Alltoallv(need_from_flat, need_from_counts, need_from_displs, MPI_INT32_T,
                  send_to_flat, send_to_counts, send_to_displs, MPI_INT32_T,
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
            send_to[p].data = send_to_flat + send_to_displs[p];
            send_to[p].size = cnt;
            send_to[p].capacity = cnt; /* view; do not free individually */
        }
    }

    /* Neighbor ranks: only those we actually communicate with. */
    Int32Vec neighbor_ranks;
    vec_init(&neighbor_ranks);
    for (int p = 0; p < comm_size; ++p)
    {
        if (need_from_counts[p] > 0 || send_to_counts[p] > 0)
        {
            vec_push(&neighbor_ranks, p);
        }
    }

    /* Per-neighbor send/recv buffers (for labels). */
    int32_t **recv_label_bufs = (int32_t **)calloc((size_t)comm_size, sizeof(int32_t *));
    int32_t **send_label_bufs = (int32_t **)calloc((size_t)comm_size, sizeof(int32_t *));
    if (!recv_label_bufs || !send_label_bufs)
    {
        fprintf(stderr, "[rank %d] Failed to allocate label buffers\n", comm_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for (int p = 0; p < comm_size; ++p)
    {
        if (need_from_counts[p] > 0)
        {
            recv_label_bufs[p] = (int32_t *)malloc((size_t)need_from_counts[p] * sizeof(int32_t));
            if (!recv_label_bufs[p])
            {
                fprintf(stderr, "[rank %d] Failed to allocate recv_label_bufs[%d]\n", comm_rank, p);
                MPI_Abort(comm, EXIT_FAILURE);
            }
        }
        if (send_to_counts[p] > 0)
        {
            send_label_bufs[p] = (int32_t *)malloc((size_t)send_to_counts[p] * sizeof(int32_t));
            if (!send_label_bufs[p])
            {
                fprintf(stderr, "[rank %d] Failed to allocate send_label_bufs[%d]\n", comm_rank, p);
                MPI_Abort(comm, EXIT_FAILURE);
            }
        }
    }

    /* OpenMP scheduling */
    const int DEFAULT_CHUNK_SIZE = 4096;

    const int chunking_enabled = (chunk_size != 1);
    const int effective_chunk = (chunk_size > 0) ? chunk_size : DEFAULT_CHUNK_SIZE;

    omp_set_schedule(chunking_enabled ? omp_sched_dynamic : omp_sched_static,
                     chunking_enabled ? effective_chunk : 0);

    /* MPI request pool for halo exchange */
    MPI_Request *requests = NULL;
    int max_reqs = 2 * neighbor_ranks.size;
    if (max_reqs > 0)
    {
        requests = (MPI_Request *)malloc((size_t)max_reqs * sizeof(MPI_Request));
        if (!requests)
        {
            fprintf(stderr, "[rank %d] Failed to allocate MPI_Request array\n", comm_rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    bool global_changed = true;
    int iterations = 0;

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
                for (int32_t li = 0; li < n_local; ++li)
                {
                    int32_t old_label = local_labels[li];
                    int32_t new_label = old_label;

                    int64_t row_begin = row_ptr[li];
                    int64_t row_end = row_ptr[li + 1];

                    for (int64_t j = row_begin; j < row_end; ++j)
                    {
                        int32_t v = col_idx[j];
                        if (v < 0 || v >= n_global)
                            continue;

                        int32_t neighbor_label;
                        if (v >= v_start && v < v_end)
                        {
                            /* Local neighbor */
                            neighbor_label = local_labels[v - v_start];
                        }
                        else
                        {
                            int32_t gidx = ghost_index[v];
                            if (gidx >= 0)
                                neighbor_label = ghost_labels[gidx];
                            else
                                neighbor_label = v; /* fallback, shouldn't happen */
                        }

                        if (neighbor_label < new_label)
                            new_label = neighbor_label;
                    }

                    local_labels_new[li] = new_label;
                    if (new_label < old_label)
                        local_changed = true;
                }

                /* Swap local label buffers */
                int32_t *tmp = local_labels;
                local_labels = local_labels_new;
                local_labels_new = tmp;
            }

            if (local_changed)
                local_changed_epoch = true;

            ++iterations;
        }

        /* ---- Halo exchange after the epoch ---- */

        /* Save old ghost labels to detect changes due to communication. */
        bool ghosts_changed = false;
        if (ghost_count > 0)
        {
            memcpy(ghost_labels_prev, ghost_labels, (size_t)ghost_count * sizeof(int32_t));
        }

        int req_count = 0;

        /* Prepare send buffers */
        for (int ni = 0; ni < neighbor_ranks.size; ++ni)
        {
            int p = neighbor_ranks.data[ni];
            int cnt = send_to_counts[p];
            if (cnt <= 0)
                continue;

            int32_t *verts = send_to[p].data;
            int32_t *buf = send_label_bufs[p];

            for (int i = 0; i < cnt; ++i)
            {
                int32_t v = verts[i];
                int32_t li = v - v_start;
                if (li >= 0 && li < n_local)
                    buf[i] = local_labels[li];
                else
                    buf[i] = v; /* safety fallback */
            }
        }

        /* Post receives */
        for (int ni = 0; ni < neighbor_ranks.size; ++ni)
        {
            int p = neighbor_ranks.data[ni];
            int cnt = need_from_counts[p];
            if (cnt <= 0)
                continue;

            MPI_Irecv(recv_label_bufs[p], cnt, MPI_INT32_T,
                      p, LABEL_TAG, comm, &requests[req_count++]);
        }

        /* Post sends */
        for (int ni = 0; ni < neighbor_ranks.size; ++ni)
        {
            int p = neighbor_ranks.data[ni];
            int cnt = send_to_counts[p];
            if (cnt <= 0)
                continue;

            MPI_Isend(send_label_bufs[p], cnt, MPI_INT32_T,
                      p, LABEL_TAG, comm, &requests[req_count++]);
        }

        if (req_count > 0)
        {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }

        /* Update ghost labels from received data. */
        if (ghost_count > 0)
        {
            for (int ni = 0; ni < neighbor_ranks.size; ++ni)
            {
                int p = neighbor_ranks.data[ni];
                int cnt = need_from_counts[p];
                if (cnt <= 0)
                    continue;

                int32_t *verts = need_from[p].data;
                int32_t *buf = recv_label_bufs[p];

                for (int i = 0; i < cnt; ++i)
                {
                    int32_t v = verts[i];
                    int32_t idx = ghost_index[v];
                    if (idx >= 0)
                        ghost_labels[idx] = buf[i];
                }
            }

            /* Check if any ghost label changed due to this exchange. */
            for (int32_t i = 0; i < ghost_count; ++i)
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
        fprintf(stderr, "CC MPI+OMP (halo, interval=%d) converged in %d iterations\n",
                eff_exchange, iterations);
    }

    /* ---- Gather global membership vector on all ranks ---- */
    int32_t base, rem;
    base = n_global / comm_size;
    rem = n_global % comm_size;

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
        int32_t ln = (r < rem) ? (base + 1) : base;
        all_counts[r] = ln;
        all_displs[r] = offset;
        offset += ln;
    }

    MPI_Allgatherv(local_labels, n_local, MPI_INT32_T,
                   labels_global, all_counts, all_displs, MPI_INT32_T,
                   comm);

    /* ---- Cleanup ---- */
    free(all_counts);
    free(all_displs);

    if (requests)
        free(requests);

    for (int p = 0; p < comm_size; ++p)
    {
        vec_free(&need_from[p]);
    }
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
    free(ghost_index);

    if (local_labels)
        free(local_labels);
    if (local_labels_new)
        free(local_labels_new);

    for (int p = 0; p < comm_size; ++p)
    {
        if (recv_label_bufs[p])
            free(recv_label_bufs[p]);
        if (send_label_bufs[p])
            free(send_label_bufs[p]);
    }
    free(recv_label_bufs);
    free(send_label_bufs);

    vec_free(&neighbor_ranks);
}

uint32_t count_connected_components(const int32_t *restrict labels_global, int32_t n_global)
{
    if (n_global <= 0 || labels_global == NULL)
        return 0;

    /* Use a bitmap to track unique component representatives */
    unsigned char *comp_bitmap = (unsigned char *)calloc((size_t)n_global, sizeof(unsigned char));
    if (!comp_bitmap)
    {
        fprintf(stderr, "count_connected_components: Failed to allocate bitmap\n");
        return 0;
    }

    uint32_t comp_count = 0;

    for (int32_t v = 0; v < n_global; ++v)
    {
        int32_t rep = labels_global[v];
        if (rep < 0 || rep >= n_global)
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