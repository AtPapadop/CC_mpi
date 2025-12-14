#define _POSIX_C_SOURCE 200112L
#include "exchange_plan.h"

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif

static void die_abort(MPI_Comm comm, const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(comm, EXIT_FAILURE);
}

/* small growable vector (uint32) */
typedef struct
{
    uint32_t *data;
    int size, cap;
} U32Vec;
static void vinit(U32Vec *v)
{
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}
static void vfree(U32Vec *v)
{
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}
static void vpush(U32Vec *v, uint32_t x, MPI_Comm comm)
{
    if (v->size == v->cap)
    {
        int nc = v->cap ? 2 * v->cap : 1024;
        uint32_t *p = (uint32_t *)realloc(v->data, (size_t)nc * sizeof(uint32_t));
        if (!p)
            die_abort(comm, "exchangeplan: OOM");
        v->data = p;
        v->cap = nc;
    }
    v->data[v->size++] = x;
}

static int *ensure_aligned_int(int *ptr, int *cap, int need)
{
    if (*cap >= need) return ptr;

    if (ptr) free(ptr);

    int *p = NULL;
    if (posix_memalign((void**)&p, 64, (size_t)need * sizeof(int)) != 0)
        return NULL;

    *cap = need;
    return p;
}

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

    /* Build need_from lists:
       need_from[p] holds vertex ids
       need_gidx[p] holds the corresponding local ghost index gi */
    U32Vec *need_from = (U32Vec *)malloc((size_t)size * sizeof(U32Vec));
    U32Vec *need_gidx = (U32Vec *)malloc((size_t)size * sizeof(U32Vec));
    if (!need_from || !need_gidx)
        die_abort(comm, "exchangeplan: OOM need_from/need_gidx");
    for (int p = 0; p < size; ++p)
    {
        vinit(&need_from[p]);
        vinit(&need_gidx[p]);
    }

    for (uint32_t gi = 0; gi < ghost_count; ++gi)
    {
        uint32_t v = ghost_vertices[gi];
        int owner = owner_fn(v, n_global, size);
        if (owner != rank)
        {
            vpush(&need_from[owner], v, comm);
            vpush(&need_gidx[owner], gi, comm);
        }
    }

    P->need_from_counts = (int *)malloc((size_t)size * sizeof(int));
    P->send_to_counts = (int *)malloc((size_t)size * sizeof(int));
    P->need_from_displs = (int *)malloc((size_t)size * sizeof(int));
    P->send_to_displs = (int *)malloc((size_t)size * sizeof(int));
    if (!P->need_from_counts || !P->send_to_counts || !P->need_from_displs || !P->send_to_displs)
        die_abort(comm, "exchangeplan: OOM counts/displs");

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
            die_abort(comm, "exchangeplan: OOM need_from_flat/gidx_flat");

        /* Flatten in the same order for both arrays */
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
            die_abort(comm, "exchangeplan: OOM send_to_flat");
    }

    MPI_Alltoallv(P->need_from_flat, P->need_from_counts, P->need_from_displs, MPI_UINT32_T,
                  P->send_to_flat, P->send_to_counts, P->send_to_displs, MPI_UINT32_T,
                  comm);

    /* active neighbor sets */
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
        die_abort(comm, "exchangeplan: OOM sources/dests");

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
        /* Avoid MPI reading from NULL even if count=0 */
        int dummy = 0;
        const int *srcs = (P->indegree > 0) ? P->sources : &dummy;
        const int *dsts = (P->outdegree > 0) ? P->dests : &dummy;

        MPI_Dist_graph_create_adjacent(comm,
                                       P->indegree, srcs, MPI_UNWEIGHTED,
                                       P->outdegree, dsts, MPI_UNWEIGHTED,
                                       MPI_INFO_NULL, 0, &P->comm_graph);
    }

    /* neighbor layouts */
    P->sendcounts = (P->outdegree > 0) ? (int *)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    P->sdispls = (P->outdegree > 0) ? (int *)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    P->recvcounts = (P->indegree > 0) ? (int *)malloc((size_t)P->indegree * sizeof(int)) : NULL;
    P->rdispls = (P->indegree > 0) ? (int *)malloc((size_t)P->indegree * sizeof(int)) : NULL;

    if ((P->outdegree > 0 && (!P->sendcounts || !P->sdispls)) ||
        (P->indegree > 0 && (!P->recvcounts || !P->rdispls)))
        die_abort(comm, "exchangeplan: OOM neighbor layouts");

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
            die_abort(comm, "exchangeplan: OOM send_labels_flat");
    }
    if (P->total_need_from > 0)
    {
        P->recv_labels_flat = (uint32_t *)malloc((size_t)P->total_need_from * sizeof(uint32_t));
        if (!P->recv_labels_flat)
            die_abort(comm, "exchangeplan: OOM recv_labels_flat");
    }

    for (int p = 0; p < size; ++p)
    {
        vfree(&need_from[p]);
        vfree(&need_gidx[p]);
    }
    free(need_from);
    free(need_gidx);

    return 0;
}

static inline uint64_t pack_pair_u32(uint32_t pos, uint32_t lbl)
{
    return (((uint64_t)pos) << 32) | (uint64_t)lbl;
}
static inline uint32_t pair_pos(uint64_t x){ return (uint32_t)(x >> 32); }
static inline uint32_t pair_lbl(uint64_t x){ return (uint32_t)(x & 0xFFFFFFFFu); }


void exchangeplan_exchange(ExchangePlan *P, uint32_t *ghost_labels, MPI_Comm comm)
{
    (void)comm;
    if (!P)
        return;

    if (P->comm_graph != MPI_COMM_NULL &&
        (P->total_send_to > 0 || P->total_need_from > 0))
    {
        MPI_Neighbor_alltoallv(P->send_labels_flat, P->sendcounts, P->sdispls, MPI_UINT32_T,
                               P->recv_labels_flat, P->recvcounts, P->rdispls, MPI_UINT32_T,
                               P->comm_graph);
    }

    for (int i = 0; i < P->total_need_from; ++i)
    {
        uint32_t gidx = P->need_from_gidx_flat[i]; /* NOW VALID */
        ghost_labels[gidx] = P->recv_labels_flat[i];
    }
}

void exchangeplan_exchange_delta(const ExchangePlan *P,
                                 const uint32_t *comp_label,
                                 const uint32_t *comp_of,
                                 uint32_t v_start, uint32_t v_end,
                                 uint32_t *prev_sent,    /* length P->total_send_to */
                                 uint32_t *ghost_labels, /* length ghost_count */
                                 MPI_Comm comm,
                                 uint64_t **sendbuf_io, int *sendcap_io,
                                 uint64_t **recvbuf_io, int *recvcap_io)
{
    if (!P || P->comm_graph == MPI_COMM_NULL)
        return;

    /* counts per neighbor (in neighbor order) */
    int outd = P->outdegree;
    int ind = P->indegree;

    int *sendcounts = NULL;
    int *sdispls = NULL;
    int *recvcounts = NULL;
    int *rdispls = NULL;

    if (outd > 0)
    {
        sendcounts = ensure_aligned_int(sendcounts, sendcap_io, outd);
        sdispls = ensure_aligned_int(sdispls, sendcap_io, outd);
        if (!sendcounts || !sdispls)
            die_abort(comm, "exchangeplan_exchange_delta: OOM sendcounts/sdispls");
    }
    if (ind > 0)
    {
        recvcounts = ensure_aligned_int(recvcounts, recvcap_io, ind);
        rdispls = ensure_aligned_int(rdispls, recvcap_io, ind);
        if (!recvcounts || !rdispls)
            die_abort(comm, "exchangeplan_exchange_delta: OOM recvcounts/rdispls");
    }
    int total_send_pairs = 0;

    /* 1) Build sendbuf sequentially in dest-neighbor order */
    if (outd > 0)
    {
        /* ensure send buffer (grow if needed) */
        /* worst-case pairs == total_send_to */
        int worst = P->total_send_to;
        if (*sendcap_io < worst)
        {
            int newcap = worst;
            uint64_t *nb = (uint64_t *)realloc(*sendbuf_io, (size_t)newcap * sizeof(uint64_t));
            if (!nb)
                die_abort(comm, "OOM: delta sendbuf");
            *sendbuf_io = nb;
            *sendcap_io = newcap;
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

                /* only send if decreases */
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

    /* 2) Exchange counts */
    if (ind > 0)
    {
        MPI_Neighbor_alltoall(sendcounts, 1, MPI_INT,
                              recvcounts, 1, MPI_INT,
                              P->comm_graph);
    }

    /* 3) Build rdispls + total recv */
    int total_recv_pairs = 0;
    for (int k = 0; k < ind; ++k)
    {
        rdispls[k] = total_recv_pairs;
        total_recv_pairs += recvcounts[k];
    }

    /* ensure recv buffer */
    if (total_recv_pairs > 0)
    {
        if (*recvcap_io < total_recv_pairs)
        {
            int newcap = total_recv_pairs;
            uint64_t *nb = (uint64_t *)realloc(*recvbuf_io, (size_t)newcap * sizeof(uint64_t));
            if (!nb)
                die_abort(comm, "OOM: delta recvbuf");
            *recvbuf_io = nb;
            *recvcap_io = newcap;
        }
    }

    /* 4) Exchange pairs */
    if (total_send_pairs > 0 || total_recv_pairs > 0)
    {
        MPI_Neighbor_alltoallv(*sendbuf_io, sendcounts, sdispls, MPI_UINT64_T,
                               *recvbuf_io, recvcounts, rdispls, MPI_UINT64_T,
                               P->comm_graph);
    }

    /* 5) Apply incoming updates */
    if (total_recv_pairs > 0)
    {
        uint64_t *rb = *recvbuf_io;
        for (int k = 0; k < ind; ++k)
        {
            int src = P->sources[k];
            int base = P->need_from_displs[src]; /* start of this source block in need_from_flat */

            int off = rdispls[k];
            int cnt = recvcounts[k];

            for (int i = 0; i < cnt; ++i)
            {
                uint64_t pr = rb[off + i];
                uint32_t pos = pair_pos(pr);
                uint32_t lbl = pair_lbl(pr);

                /* receiver’s need_from order matches sender’s send_to order */
                int global_pos = base + (int)pos;
                uint32_t gidx = P->need_from_gidx_flat[global_pos];
                ghost_labels[gidx] = lbl;
            }
        }
    }
}
