#define _POSIX_C_SOURCE 200112L

#include "graph_dist.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <limits.h>

#include "graph.h"

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif



static void compute_vertex_range(uint32_t n, int comm_size, int rank,
                                 uint32_t *v_start, uint32_t *v_end)
{
    if (comm_size <= 0)
    {
        *v_start = 0;
        *v_end = 0;
        return;
    }

    uint32_t comm_size_u = (uint32_t)comm_size;
    uint32_t rank_u = (uint32_t)rank;
    uint32_t base = (comm_size_u > 0) ? (n / comm_size_u) : 0;
    uint32_t rem = (comm_size_u > 0) ? (n % comm_size_u) : 0;

    if (rank_u < rem)
    {
        uint32_t local_n = base + 1u;
        *v_start = rank_u * local_n;
        *v_end = *v_start + local_n;
    }
    else
    {
        uint32_t local_n = base;
        uint32_t extra = (rem > 0) ? rem * (base + 1u) : 0;
        uint32_t offset = (rank_u > rem) ? (rank_u - rem) * base : 0;
        *v_start = extra + offset;
        *v_end = *v_start + local_n;
    }
}


static void compute_weighted_bounds_from_rowptr(const uint64_t *row_ptr,
                                                uint32_t n,
                                                int comm_size,
                                                uint32_t vertex_weight,
                                                uint32_t *bounds)
{
    if (comm_size <= 0) return;

    const uint64_t total_edges = row_ptr ? row_ptr[n] : 0;
    const uint64_t total_w = total_edges + (uint64_t)vertex_weight * (uint64_t)n;

    bounds[0] = 0;

    uint32_t v = 0;
    uint64_t cum_w = 0;

    for (int r = 1; r < comm_size; ++r)
    {
        uint64_t target = (total_w * (uint64_t)r) / (uint64_t)comm_size;

        while (v < n && cum_w < target)
        {
            uint64_t deg = row_ptr[v + 1] - row_ptr[v];
            cum_w += deg + (uint64_t)vertex_weight;
            v++;
        }

        uint32_t min_v = bounds[r - 1] + 1u;
        uint32_t max_v = n - (uint32_t)(comm_size - r);
        if (v < min_v) v = min_v;
        if (v > max_v) v = max_v;

        bounds[r] = v;
    }

    bounds[comm_size] = n;
}


static void scatter_col_idx_large(const uint32_t *sendbuf,
                                  const uint64_t *sendcounts,
                                  const uint64_t *displs,
                                  uint32_t *recvbuf,
                                  uint64_t recvcount,
                                  MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const uint64_t chunk_limit = (uint64_t)INT_MAX;

    if (rank == 0)
    {
        for (int r = 0; r < size; ++r)
        {
            uint64_t count = sendcounts ? sendcounts[r] : 0;
            uint64_t offset = displs ? displs[r] : 0;

            if (r == 0)
            {
                if (count > 0 && recvbuf)
                    memcpy(recvbuf, sendbuf + offset, (size_t)count * sizeof(uint32_t));
                continue;
            }

            uint64_t sent = 0;
            while (sent < count)
            {
                int chunk = (int)((count - sent) > chunk_limit ? chunk_limit : (count - sent));
                MPI_Send(sendbuf + offset + sent, chunk, MPI_UINT32_T, r, 0, comm);
                sent += (uint64_t)chunk;
            }
        }
    }
    else
    {
        uint64_t received = 0;
        while (received < recvcount)
        {
            int chunk = (int)((recvcount - received) > chunk_limit ? chunk_limit : (recvcount - received));
            MPI_Recv(recvbuf + received, chunk, MPI_UINT32_T, 0, 0, comm, MPI_STATUS_IGNORE);
            received += (uint64_t)chunk;
        }
    }
}

static int load_dist_csr_from_file_rank0(const char *path,
                                         int symmetrize,
                                         int drop_self_loops,
                                         int use_edge,
                                         DistCSRGraph *out,
                                         MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    CSRGraph full;
    int rc = 0;

    if (rank == 0)
        rc = load_csr_from_file(path, symmetrize, drop_self_loops, &full);

    MPI_Bcast(&rc, 1, MPI_INT, 0, comm);
    if (rc != 0)
    {
        if (rank == 0)
            free_csr(&full);
        return rc;
    }

    uint32_t n_global = 0;
    uint64_t m_global = 0;
    if (rank == 0)
    {
        n_global = full.n;
        m_global = full.m;
    }
    MPI_Bcast(&n_global, 1, MPI_UINT32_T, 0, comm);
    MPI_Bcast(&m_global, 1, MPI_UINT64_T, 0, comm);

    uint32_t v_start = 0, v_end = 0;

    out->part_bounds = NULL;
    out->part_size = 0;
    out->part_kind = 0;

    if (use_edge && size > 1)
    {
        uint32_t *bounds = (uint32_t *)malloc((size_t)(size + 1) * sizeof(uint32_t));
        if (!bounds)
        {
            fprintf(stderr, "[rank %d] Failed to allocate bounds\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        if (rank == 0)
        {
            const uint32_t vertex_weight = 1;
            compute_weighted_bounds_from_rowptr(full.row_ptr, n_global, size, vertex_weight, bounds);
        }

        MPI_Bcast(bounds, size + 1, MPI_UINT32_T, 0, comm);

        v_start = bounds[rank];
        v_end = bounds[rank + 1];

        out->part_bounds = bounds;
        out->part_size = size;
        out->part_kind = 1;
    }
    else
    {
        compute_vertex_range(n_global, size, rank, &v_start, &v_end);
    }

    uint32_t n_local = v_end - v_start;

    int *sendcounts_rowptr = NULL;
    int *displs_rowptr = NULL;

    if (rank == 0)
    {
        sendcounts_rowptr = (int *)malloc((size_t)size * sizeof(int));
        displs_rowptr = (int *)malloc((size_t)size * sizeof(int));
        if (!sendcounts_rowptr || !displs_rowptr)
        {
            fprintf(stderr, "Failed to allocate sendcounts_rowptr/displs_rowptr\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }

        for (int r = 0; r < size; ++r)
        {
            uint32_t s_v_start, s_v_end;
            if (use_edge && out->part_kind == 1)
            {
                s_v_start = out->part_bounds[r];
                s_v_end = out->part_bounds[r + 1];
            }
            else
            {
                compute_vertex_range(n_global, size, r, &s_v_start, &s_v_end);
            }

            uint32_t s_n_local = s_v_end - s_v_start;
            sendcounts_rowptr[r] = (int)(s_n_local + 1u);
            displs_rowptr[r] = (int)s_v_start;
        }
    }

    uint64_t *local_row_ptr = (uint64_t *)malloc((size_t)(n_local + 1) * sizeof(uint64_t));
    if (!local_row_ptr)
    {
        fprintf(stderr, "[rank %d] Failed to allocate local_row_ptr\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    MPI_Scatterv(rank == 0 ? full.row_ptr : NULL,
                 sendcounts_rowptr,
                 displs_rowptr,
                 MPI_UINT64_T,
                 local_row_ptr,
                 (int)(n_local + 1u),
                 MPI_UINT64_T,
                 0,
                 comm);

    if (rank == 0)
    {
        free(sendcounts_rowptr);
        free(displs_rowptr);
    }

    uint64_t base_edge = (n_local > 0) ? local_row_ptr[0] : 0;
    for (uint32_t i = 0; i <= n_local; ++i)
        local_row_ptr[i] -= base_edge;

    uint64_t m_local = (n_local > 0) ? local_row_ptr[n_local] : 0;

    uint64_t *sendcounts_colidx = NULL;
    uint64_t *displs_colidx = NULL;

    if (rank == 0)
    {
        sendcounts_colidx = (uint64_t *)malloc((size_t)size * sizeof(uint64_t));
        displs_colidx = (uint64_t *)malloc((size_t)size * sizeof(uint64_t));
        if (!sendcounts_colidx || !displs_colidx)
        {
            fprintf(stderr, "Failed to allocate sendcounts_colidx/displs_colidx\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }

        for (int r = 0; r < size; ++r)
        {
            uint32_t s_v_start, s_v_end;
            if (use_edge && out->part_kind == 1)
            {
                s_v_start = out->part_bounds[r];
                s_v_end = out->part_bounds[r + 1];
            }
            else
            {
                compute_vertex_range(n_global, size, r, &s_v_start, &s_v_end);
            }
            uint64_t row_begin = full.row_ptr[s_v_start];
            uint64_t row_end = full.row_ptr[s_v_end];
            sendcounts_colidx[r] = row_end - row_begin;
            displs_colidx[r] = row_begin;
        }
    }

    uint32_t *local_col_idx = NULL;
    if (m_local > 0)
    {
        local_col_idx = (uint32_t *)malloc((size_t)m_local * sizeof(uint32_t));
        if (!local_col_idx)
        {
            fprintf(stderr, "[rank %d] Failed to allocate local_col_idx\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    scatter_col_idx_large(rank == 0 ? full.col_idx : NULL,
                          sendcounts_colidx,
                          displs_colidx,
                          local_col_idx,
                          m_local,
                          comm);

    if (rank == 0)
    {
        free(sendcounts_colidx);
        free(displs_colidx);
        free_csr(&full);
    }

    out->n_global = n_global;
    out->m_global = m_global;
    out->n_local = n_local;
    out->m_local = m_local;
    out->v_start = v_start;
    out->v_end = v_end;
    out->row_ptr = local_row_ptr;
    out->col_idx = local_col_idx;

    return 0;
}

int load_dist_csr_from_file(const char *path,
                            int symmetrize,
                            int drop_self_loops,
                            int use_edge,
                            DistCSRGraph *out,
                            MPI_Comm comm)
{
    const char *ext = strrchr(path, '.');
    if (ext && strcasecmp(ext, ".mat") == 0)
    {
        return load_dist_csr_from_file_rank0(path, symmetrize, drop_self_loops, use_edge, out, comm);
    }

    return -1;
}

void free_dist_csr(DistCSRGraph *g)
{
    if (!g)
        return;
    free(g->row_ptr);
    free(g->col_idx);
    free(g->part_bounds);
    g->row_ptr = NULL;
    g->col_idx = NULL;
    g->part_bounds = NULL;
    g->part_size = 0;
    g->part_kind = 0;
    g->n_global = 0;
    g->m_global = 0;
    g->n_local = 0;
    g->m_local = 0;
    g->v_start = 0;
    g->v_end = 0;
}
