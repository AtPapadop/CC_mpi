#define _POSIX_C_SOURCE 200112L
#include "exchange_plan.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "runtime_utils.h"
#include "vec_helpers.h"

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif

void exchangeplan_free(ExchangePlan *P)
{
    if (!P)
        return;

    free(P->sources);
    free(P->dests);

    free(P->need_from_counts);
    free(P->send_to_counts);
    free(P->need_from_displs);
    free(P->send_to_displs);

    free(P->need_from_flat);
    free(P->need_from_gidx_flat);
    free(P->send_to_flat);

    free(P->sendcounts);
    free(P->sdispls);
    free(P->recvcounts);
    free(P->rdispls);

    free(P->send_labels_flat);
    free(P->recv_labels_flat);

    memset(P, 0, sizeof(*P));
    P->comm_graph = MPI_COMM_NULL;
}

int exchangeplan_build(ExchangePlan *P,
                       const uint32_t *ghost_vertices,
                       uint32_t ghost_count,
                       uint32_t n_global,
                       MPI_Comm comm,
                       OwnerFn owner_fn)
{
    if (!P)
        return 1;
    exchangeplan_free(P);

    MPI_Comm_rank(comm, &P->comm_rank);
    MPI_Comm_size(comm, &P->comm_size);
    const int rank = P->comm_rank;
    const int size = P->comm_size;

    U32VecI *need_from = (U32VecI *)malloc((size_t)size * sizeof(U32VecI));
    U32VecI *need_gidx = (U32VecI *)malloc((size_t)size * sizeof(U32VecI));
    if (!need_from || !need_gidx)
        mpi_die_abort(comm, "exchangeplan: OOM need_from/need_gidx");
    for (int p = 0; p < size; ++p)
    {
        u32veci_init(&need_from[p]);
        u32veci_init(&need_gidx[p]);
    }

    for (uint32_t gi = 0; gi < ghost_count; ++gi)
    {
        uint32_t v = ghost_vertices[gi];
        int owner = owner_fn(v, n_global, size);
        if (owner != rank)
        {
                u32veci_push(&need_from[owner], v, comm);
                u32veci_push(&need_gidx[owner], gi, comm);
        }
    }

    P->need_from_counts = (int *)malloc((size_t)size * sizeof(int));
    P->send_to_counts = (int *)malloc((size_t)size * sizeof(int));
    P->need_from_displs = (int *)malloc((size_t)size * sizeof(int));
    P->send_to_displs = (int *)malloc((size_t)size * sizeof(int));
    if (!P->need_from_counts || !P->send_to_counts || !P->need_from_displs || !P->send_to_displs)
        mpi_die_abort(comm, "exchangeplan: OOM counts/displs");

    for (int p = 0; p < size; ++p)
        P->need_from_counts[p] = need_from[p].size;

    MPI_Alltoall(P->need_from_counts, 1, MPI_INT,
                 P->send_to_counts, 1, MPI_INT, comm);

    int total_need = 0, total_send = 0;
    for (int p = 0; p < size; ++p)
    {
        P->need_from_displs[p] = total_need;
        total_need += P->need_from_counts[p];
    }
    for (int p = 0; p < size; ++p)
    {
        P->send_to_displs[p] = total_send;
        total_send += P->send_to_counts[p];
    }

    P->total_need_from = total_need;
    P->total_send_to = total_send;

    if (total_need > 0)
    {
        P->need_from_flat = (uint32_t *)malloc((size_t)total_need * sizeof(uint32_t));
        P->need_from_gidx_flat = (uint32_t *)malloc((size_t)total_need * sizeof(uint32_t));
        if (!P->need_from_flat || !P->need_from_gidx_flat)
            mpi_die_abort(comm, "exchangeplan: OOM need_from_flat/gidx_flat");

        for (int p = 0; p < size; ++p)
        {
            int cnt = need_from[p].size;
            if (cnt > 0)
            {
                memcpy(P->need_from_flat + P->need_from_displs[p],
                       need_from[p].data, (size_t)cnt * sizeof(uint32_t));
                memcpy(P->need_from_gidx_flat + P->need_from_displs[p],
                       need_gidx[p].data, (size_t)cnt * sizeof(uint32_t));
            }
        }
    }

    if (total_send > 0)
    {
        P->send_to_flat = (uint32_t *)malloc((size_t)total_send * sizeof(uint32_t));
        if (!P->send_to_flat)
            mpi_die_abort(comm, "exchangeplan: OOM send_to_flat");
    }

    MPI_Alltoallv(P->need_from_flat, P->need_from_counts, P->need_from_displs, MPI_UINT32_T,
                  P->send_to_flat, P->send_to_counts, P->send_to_displs, MPI_UINT32_T,
                  comm);

    P->indegree = 0;
    P->outdegree = 0;
    for (int p = 0; p < size; ++p)
    {
        if (P->need_from_counts[p] > 0)
            ++P->indegree;
        if (P->send_to_counts[p] > 0)
            ++P->outdegree;
    }

    P->sources = (P->indegree > 0) ? (int *)malloc((size_t)P->indegree * sizeof(int)) : NULL;
    P->dests = (P->outdegree > 0) ? (int *)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    if ((P->indegree > 0 && !P->sources) || (P->outdegree > 0 && !P->dests))
        mpi_die_abort(comm, "exchangeplan: OOM sources/dests");

    int ii = 0, oo = 0;
    for (int p = 0; p < size; ++p)
    {
        if (P->need_from_counts[p] > 0)
            P->sources[ii++] = p;
        if (P->send_to_counts[p] > 0)
            P->dests[oo++] = p;
    }

    P->comm_graph = MPI_COMM_NULL;
    if (P->indegree > 0 || P->outdegree > 0)
    {
        int dummy = 0;
        const int *srcs = (P->indegree > 0) ? P->sources : &dummy;
        const int *dsts = (P->outdegree > 0) ? P->dests : &dummy;

        MPI_Dist_graph_create_adjacent(comm,
                                       P->indegree, srcs, MPI_UNWEIGHTED,
                                       P->outdegree, dsts, MPI_UNWEIGHTED,
                                       MPI_INFO_NULL, 0, &P->comm_graph);
    }

    P->sendcounts = (P->outdegree > 0) ? (int *)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    P->sdispls = (P->outdegree > 0) ? (int *)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    P->recvcounts = (P->indegree > 0) ? (int *)malloc((size_t)P->indegree * sizeof(int)) : NULL;
    P->rdispls = (P->indegree > 0) ? (int *)malloc((size_t)P->indegree * sizeof(int)) : NULL;

    if ((P->outdegree > 0 && (!P->sendcounts || !P->sdispls)) ||
        (P->indegree > 0 && (!P->recvcounts || !P->rdispls)))
        mpi_die_abort(comm, "exchangeplan: OOM neighbor layouts");

    for (int k = 0; k < P->outdegree; ++k)
    {
        int p = P->dests[k];
        P->sendcounts[k] = P->send_to_counts[p];
        P->sdispls[k] = P->send_to_displs[p];
    }
    for (int k = 0; k < P->indegree; ++k)
    {
        int p = P->sources[k];
        P->recvcounts[k] = P->need_from_counts[p];
        P->rdispls[k] = P->need_from_displs[p];
    }

    if (P->total_send_to > 0)
    {
        P->send_labels_flat = (uint32_t *)malloc((size_t)P->total_send_to * sizeof(uint32_t));
        if (!P->send_labels_flat)
            mpi_die_abort(comm, "exchangeplan: OOM send_labels_flat");
    }
    if (P->total_need_from > 0)
    {
        P->recv_labels_flat = (uint32_t *)malloc((size_t)P->total_need_from * sizeof(uint32_t));
        if (!P->recv_labels_flat)
            mpi_die_abort(comm, "exchangeplan: OOM recv_labels_flat");
    }

    for (int p = 0; p < size; ++p)
    {
        u32veci_free(&need_from[p]);
        u32veci_free(&need_gidx[p]);
    }
    free(need_from);
    free(need_gidx);

    return 0;
}

static inline uint64_t pack_pair_u32(uint32_t pos, uint32_t lbl)
{
    return (((uint64_t)pos) << 32) | (uint64_t)lbl;
}
static inline uint32_t pair_pos(uint64_t x) { return (uint32_t)(x >> 32); }
static inline uint32_t pair_lbl(uint64_t x) { return (uint32_t)(x & 0xFFFFFFFFu); }

void exchangeplan_exchange(ExchangePlan *P, uint32_t *ghost_labels, MPI_Comm comm)
{
    (void)comm;
    MPI_Request req;
    exchangeplan_exchange_start(P, &req);
    exchangeplan_exchange_finish(P, ghost_labels, &req);
}

void exchangeplan_exchange_delta(const ExchangePlan *P,
                                 const uint32_t *comp_label,
                                 const uint32_t *comp_of,
                                 uint32_t v_start, uint32_t v_end,
                                 uint32_t *prev_sent,
                                 uint32_t *ghost_labels,
                                 MPI_Comm comm,
                                 uint64_t **sendbuf_io, int *sendcap_io,
                                 uint64_t **recvbuf_io, int *recvcap_io)
{
    if (!P || P->comm_graph == MPI_COMM_NULL)
        return;

    const int outd = P->outdegree;
    const int ind = P->indegree;

    int dummy_i = 0;
    uint64_t dummy_u64 = 0;

    int *sendcounts = NULL, *sdispls = NULL;
    int *recvcounts = NULL, *rdispls = NULL;

    if (outd > 0)
    {
        sendcounts = (int *)malloc((size_t)outd * sizeof(int));
        sdispls = (int *)malloc((size_t)outd * sizeof(int));
        if (!sendcounts || !sdispls)
            mpi_die_abort(comm, "exchangeplan_exchange_delta: OOM sendcounts/sdispls");
        memset(sendcounts, 0, (size_t)outd * sizeof(int));
        memset(sdispls, 0, (size_t)outd * sizeof(int));
    }
    else
    {
        sendcounts = &dummy_i;
        sdispls = &dummy_i;
    }

    if (ind > 0)
    {
        recvcounts = (int *)malloc((size_t)ind * sizeof(int));
        rdispls = (int *)malloc((size_t)ind * sizeof(int));
        if (!recvcounts || !rdispls)
            mpi_die_abort(comm, "exchangeplan_exchange_delta: OOM recvcounts/rdispls");
        memset(recvcounts, 0, (size_t)ind * sizeof(int));
        memset(rdispls, 0, (size_t)ind * sizeof(int));
    }
    else
    {
        recvcounts = &dummy_i;
        rdispls = &dummy_i;
    }

    if (*sendbuf_io == NULL)
    {
        *sendbuf_io = (uint64_t *)malloc(sizeof(uint64_t));
        if (!*sendbuf_io)
            mpi_die_abort(comm, "OOM: delta sendbuf init");
        *sendcap_io = 1;
    }
    if (*recvbuf_io == NULL)
    {
        *recvbuf_io = (uint64_t *)malloc(sizeof(uint64_t));
        if (!*recvbuf_io)
            mpi_die_abort(comm, "OOM: delta recvbuf init");
        *recvcap_io = 1;
    }

    int total_send_pairs = 0;

    if (outd > 0)
    {
        const int worst = P->total_send_to;
        if (*sendcap_io < worst)
        {
            uint64_t *nb = (uint64_t *)realloc(*sendbuf_io, (size_t)worst * sizeof(uint64_t));
            if (!nb)
                mpi_die_abort(comm, "OOM: delta sendbuf grow");
            *sendbuf_io = nb;
            *sendcap_io = worst;
        }

        uint64_t *sendbuf = *sendbuf_io;

        for (int k = 0; k < outd; ++k)
        {
            int dest = P->dests[k];
            int base = P->send_to_displs[dest];
            int cnt = P->send_to_counts[dest];

            sdispls[k] = total_send_pairs;
            int wrote = 0;

            for (int i = 0; i < cnt; ++i)
            {
                int idx = base + i;
                uint32_t v = P->send_to_flat[idx];

                uint32_t lbl = v;
                if (v >= v_start && v < v_end)
                {
                    uint32_t li = v - v_start;
                    uint32_t rep = comp_of[li];
                    lbl = comp_label[rep];
                }

                if (lbl < prev_sent[idx])
                {
                    prev_sent[idx] = lbl;
                    sendbuf[total_send_pairs + wrote] = pack_pair_u32((uint32_t)i, lbl);
                    wrote++;
                }
            }

            sendcounts[k] = wrote;
            total_send_pairs += wrote;
        }
    }

    MPI_Neighbor_alltoall(sendcounts, 1, MPI_INT,
                          recvcounts, 1, MPI_INT,
                          P->comm_graph);

    int total_recv_pairs = 0;
    if (ind > 0)
    {
        for (int k = 0; k < ind; ++k)
        {
            rdispls[k] = total_recv_pairs;
            total_recv_pairs += recvcounts[k];
        }
    }

    if (total_recv_pairs > 0 && *recvcap_io < total_recv_pairs)
    {
        uint64_t *nb = (uint64_t *)realloc(*recvbuf_io, (size_t)total_recv_pairs * sizeof(uint64_t));
        if (!nb)
            mpi_die_abort(comm, "OOM: delta recvbuf grow");
        *recvbuf_io = nb;
        *recvcap_io = total_recv_pairs;
    }

    void *sb = (total_send_pairs > 0) ? (void *)(*sendbuf_io) : (void *)&dummy_u64;
    void *rb = (total_recv_pairs > 0) ? (void *)(*recvbuf_io) : (void *)&dummy_u64;

    MPI_Neighbor_alltoallv(sb, sendcounts, sdispls, MPI_UINT64_T,
                           rb, recvcounts, rdispls, MPI_UINT64_T,
                           P->comm_graph);

    if (total_recv_pairs > 0)
    {
        uint64_t *rbuf = *recvbuf_io;

        for (int k = 0; k < ind; ++k)
        {
            int src = P->sources[k];
            int base = P->need_from_displs[src];

            int off = rdispls[k];
            int cnt = recvcounts[k];

            for (int i = 0; i < cnt; ++i)
            {
                uint64_t pr = rbuf[off + i];
                uint32_t pos = pair_pos(pr);
                uint32_t lbl = pair_lbl(pr);

                int global_pos = base + (int)pos;
                uint32_t gidx = P->need_from_gidx_flat[global_pos];
                ghost_labels[gidx] = lbl;
            }
        }
    }

    if (outd > 0)
    {
        free(sendcounts);
        free(sdispls);
    }
    if (ind > 0)
    {
        free(recvcounts);
        free(rdispls);
    }
}

void exchangeplan_exchange_start(ExchangePlan *P, MPI_Request *req)
{
    if (!P || !req) return;
    *req = MPI_REQUEST_NULL;

    if (P->comm_graph != MPI_COMM_NULL &&
        (P->total_send_to > 0 || P->total_need_from > 0))
    {
        MPI_Ineighbor_alltoallv(P->send_labels_flat, P->sendcounts, P->sdispls, MPI_UINT32_T,
                                P->recv_labels_flat, P->recvcounts, P->rdispls, MPI_UINT32_T,
                                P->comm_graph, req);
    }
}

void exchangeplan_exchange_finish(ExchangePlan *P, uint32_t *ghost_labels, MPI_Request *req)
{
    if (!P || !ghost_labels || !req) return;

    if (*req != MPI_REQUEST_NULL)
        MPI_Wait(req, MPI_STATUS_IGNORE);

    for (int i = 0; i < P->total_need_from; ++i)
    {
        uint32_t gidx = P->need_from_gidx_flat[i];
        ghost_labels[gidx] = P->recv_labels_flat[i];
    }
}
