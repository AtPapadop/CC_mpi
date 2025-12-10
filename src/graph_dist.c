#define _POSIX_C_SOURCE 200112L

#include "graph_dist.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "graph.h"
#include "mmio.h"

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

/**
 * Compute the range of vertices owned by a given rank.
 * The global vertex IDs are in [0, n).
 * The vertices are distributed as evenly as possible.
 */
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

typedef struct
{
    uint32_t u, v;
} Edge;

/**
 * Comparator for qsort to sort edges by (u,v).
 */
static int cmp_edge(const void *a, const void *b)
{
    Edge *ea = (Edge *)a;
    Edge *eb = (Edge *)b;

    if (ea->u != eb->u)
        return (ea->u < eb->u) ? -1 : 1;
    else if (ea->v != eb->v)
        return (ea->v < eb->v) ? -1 : 1;
    else
        return 0;
}

/**
 * Deduplicate sorted edges in-place, updating the edge count and, optionally, removing self-loops.
 */
static void dedup_edges(Edge *E, uint64_t *m, int drop_self_loops)
{
    uint64_t write = 0;
    for (uint64_t r = 0; r < *m; ++r)
    {
        if (drop_self_loops && E[r].u == E[r].v)
            continue;
        if (write == 0 ||
            E[r].u != E[write - 1].u ||
            E[r].v != E[write - 1].v)
        {
            E[write++] = E[r];
        }
    }
    *m = write;
}

typedef struct
{
    Edge *data;
    uint64_t size;
    uint64_t capacity;
} EdgeVec;

static void edgevec_init(EdgeVec *v)
{
    v->data = NULL;
    v->size = 0;
    v->capacity = 0;
}

static void edgevec_reserve(EdgeVec *v, uint64_t new_cap)
{
    if (new_cap <= v->capacity)
        return;
    Edge *new_data = (Edge *)realloc(v->data, (size_t)new_cap * sizeof(Edge));
    if (!new_data)
    {
        fprintf(stderr, "edgevec_reserve: out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    v->data = new_data;
    v->capacity = new_cap;
}

static void edgevec_push(EdgeVec *v, Edge e)
{
    if (v->size == v->capacity)
    {
        uint64_t new_cap = (v->capacity > 0) ? 2 * v->capacity : 1024;
        edgevec_reserve(v, new_cap);
    }
    v->data[v->size++] = e;
}

static void edgevec_free(EdgeVec *v)
{
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->capacity = 0;
}

/**
 * Load a distributed CSR graph from a Matrix Market (.mtx) or text file in parallel.
 * Each rank reads the file and constructs only its own rows.
 */
static int load_dist_csr_from_mtx_parallel(const char *path,
                                           int symmetrize,
                                           int drop_self_loops,
                                           DistCSRGraph *out,
                                           MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "[rank %d] Error opening %s\n", rank, path);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Could not process Matrix Market banner.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode))
    {
        if (rank == 0)
            fprintf(stderr, "Only sparse coordinate matrices are supported.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Failed reading size line.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    uint32_t n = (uint32_t)((M > N) ? M : N);
    int symmetric_in_file =
        mm_is_symmetric(matcode) || mm_is_hermitian(matcode) || mm_is_skew(matcode);

    /* Compute local vertex block for this rank. */
    uint32_t v_start, v_end;
    compute_vertex_range(n, size, rank, &v_start, &v_end);
    uint32_t n_local = v_end - v_start;

    /* Collect edges whose row is in [v_start, v_end). */
    EdgeVec edges;
    edgevec_init(&edges);

    for (int k = 0; k < nz; ++k)
    {
        int i, j;
        if (mm_is_pattern(matcode))
        {
            if (fscanf(f, "%d %d", &i, &j) != 2)
                break;
        }
        else
        {
            double val;
            if (fscanf(f, "%d %d %lf", &i, &j, &val) < 2)
                break;
        }

        i--;
        j--; /* convert to 0-based */

        if (i < 0 || j < 0 || (uint32_t)i >= n || (uint32_t)j >= n)
            continue;

        /* Forward edge (i,j) if i in our row block. */
        if ((uint32_t)i >= v_start && (uint32_t)i < v_end)
        {
            edgevec_push(&edges, (Edge){(uint32_t)i, (uint32_t)j});
        }

        /* If symmetric or symmetrize, also add (j,i) if j in our block. */
        if ((symmetric_in_file || symmetrize) && i != j)
        {
            if ((uint32_t)j >= v_start && (uint32_t)j < v_end)
            {
                edgevec_push(&edges, (Edge){(uint32_t)j, (uint32_t)i});
            }
        }
    }

    fclose(f);

    /* Sort and deduplicate local edges. */
    uint64_t m_local = edges.size;
    if (m_local > 0)
    {
        qsort(edges.data, (size_t)m_local, sizeof(Edge), cmp_edge);
        dedup_edges(edges.data, &m_local, drop_self_loops);
    }

    /* Build local CSR: rows 0..n_local-1 correspond to global vertices v_start..v_end-1. */
    uint64_t *row_ptr = (uint64_t *)calloc((size_t)n_local + 1, sizeof(uint64_t));
    if (!row_ptr)
    {
        fprintf(stderr, "[rank %d] Failed to allocate row_ptr\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for (uint64_t r = 0; r < m_local; ++r)
    {
        uint32_t u_global = edges.data[r].u;
        if (u_global < v_start || u_global >= v_end)
            continue; /* defensive */
        uint32_t u_local = u_global - v_start;
        row_ptr[u_local + 1]++;
    }
    for (uint32_t i = 0; i < n_local; ++i)
    {
        row_ptr[i + 1] += row_ptr[i];
    }

    uint32_t *col_idx = NULL;
    if (m_local > 0)
    {
        col_idx = (uint32_t *)malloc((size_t)m_local * sizeof(uint32_t));
        if (!col_idx)
        {
            fprintf(stderr, "[rank %d] Failed to allocate col_idx\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    uint64_t *head = (uint64_t *)malloc((size_t)n_local * sizeof(uint64_t));
    if (!head)
    {
        fprintf(stderr, "[rank %d] Failed to allocate head array\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    memcpy(head, row_ptr, (size_t)n_local * sizeof(uint64_t));

    for (uint64_t r = 0; r < m_local; ++r)
    {
        uint32_t u_global = edges.data[r].u;
        uint32_t v_global = edges.data[r].v;
        if (u_global < v_start || u_global >= v_end)
            continue;
        uint32_t u_local = u_global - v_start;

        uint64_t pos = head[u_local]++;
        col_idx[pos] = v_global; /* store neighbor as GLOBAL vertex */
    }

    free(head);
    edgevec_free(&edges);

    /* Compute global m via Allreduce. */
    uint64_t m_global = 0;
    MPI_Allreduce(&m_local, &m_global, 1, MPI_UINT64_T, MPI_SUM, comm);

    /* Fill DistCSRGraph. */
    out->n_global = n;
    out->m_global = m_global;
    out->n_local = n_local;
    out->m_local = m_local;
    out->v_start = v_start;
    out->v_end = v_end;
    out->row_ptr = row_ptr;
    out->col_idx = col_idx;

    return 0;
}


/**
 * Load distributed CSR graph in .mat format by having rank 0 load the full graph and scatter rows.
 */
static int load_dist_csr_from_file_rank0(const char *path,
                                         int symmetrize,
                                         int drop_self_loops,
                                         DistCSRGraph *out,
                                         MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    CSRGraph full;
    int rc = 0;

    if (rank == 0)
    {
        rc = load_csr_from_file(path, symmetrize, drop_self_loops, &full);
    }

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

    uint32_t v_start, v_end;
    compute_vertex_range(n_global, size, rank, &v_start, &v_end);
    uint32_t n_local = v_end - v_start;

    /* Scatter row_ptr segments: each rank gets row_ptr[v_start..v_end] (n_local+1 entries). */
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

        uint32_t s_v_start, s_v_end;
        for (int r = 0; r < size; ++r)
        {
            compute_vertex_range(n_global, size, r, &s_v_start, &s_v_end);
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
                 n_local + 1,
                 MPI_UINT64_T,
                 0,
                 comm);

    if (rank == 0)
    {
        free(sendcounts_rowptr);
        free(displs_rowptr);
    }

    /* Convert local_row_ptr from global offsets to local offsets. */
    uint64_t base_edge = (n_local > 0) ? local_row_ptr[0] : 0;
    for (uint32_t i = 0; i <= n_local; ++i)
    {
        local_row_ptr[i] -= base_edge;
    }
    uint64_t m_local = (n_local > 0) ? local_row_ptr[n_local] : 0;

    /* Scatter col_idx segments: edges belonging to rows [v_start, v_end). */
    int *sendcounts_colidx = NULL;
    int *displs_colidx = NULL;

    if (rank == 0)
    {
        sendcounts_colidx = (int *)malloc((size_t)size * sizeof(int));
        displs_colidx = (int *)malloc((size_t)size * sizeof(int));
        if (!sendcounts_colidx || !displs_colidx)
        {
            fprintf(stderr, "Failed to allocate sendcounts_colidx/displs_colidx\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }

        uint32_t s_v_start, s_v_end;
        for (int r = 0; r < size; ++r)
        {
            compute_vertex_range(n_global, size, r, &s_v_start, &s_v_end);
            uint64_t row_begin = full.row_ptr[s_v_start];
            uint64_t row_end = full.row_ptr[s_v_end];
            sendcounts_colidx[r] = (int)(row_end - row_begin);
            displs_colidx[r] = (int)row_begin;
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

    MPI_Scatterv(rank == 0 ? full.col_idx : NULL,
                 sendcounts_colidx,
                 displs_colidx,
                 MPI_UINT32_T,
                 local_col_idx,
                 (int)m_local,
                 MPI_UINT32_T,
                 0,
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

/* ----------------- Public API: load_dist_csr_from_file ----------------- */

int load_dist_csr_from_file(const char *path,
                            int symmetrize,
                            int drop_self_loops,
                            DistCSRGraph *out,
                            MPI_Comm comm)
{
    const char *ext = strrchr(path, '.');
    if (!ext)
        ext = "";

    /* Matrix Market / text: fully parallel loader (no rank-0 big CSR). */
    if (strcasecmp(ext, ".mtx") == 0 || strcasecmp(ext, ".txt") == 0)
    {
        return load_dist_csr_from_mtx_parallel(path, symmetrize, drop_self_loops, out, comm);
    }

    /* MATLAB .mat or anything else: fall back to rank-0 loader + scatter. */
    return load_dist_csr_from_file_rank0(path, symmetrize, drop_self_loops, out, comm);
}

void free_dist_csr(DistCSRGraph *g)
{
    if (!g)
        return;
    free(g->row_ptr);
    free(g->col_idx);
    g->row_ptr = NULL;
    g->col_idx = NULL;
    g->n_global = 0;
    g->m_global = 0;
    g->n_local = 0;
    g->m_local = 0;
    g->v_start = 0;
    g->v_end = 0;
}