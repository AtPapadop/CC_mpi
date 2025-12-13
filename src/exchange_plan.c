#define _POSIX_C_SOURCE 200112L

#include "exchange_plan.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

static void die_abort(MPI_Comm comm, const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(comm, EXIT_FAILURE);
}

/* lower_bound on sorted uint32_t array */
static uint32_t lower_bound_u32(const uint32_t *arr, uint32_t n, uint32_t key, int *found)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi)
    {
        uint32_t mid = lo + (hi - lo) / 2;
        uint32_t v = arr[mid];
        if (v < key) lo = mid + 1;
        else hi = mid;
    }
    if (found) *found = (lo < n && arr[lo] == key);
    return lo;
}

/* internal tiny vec for uint32_t with int size (MPI counts) */
typedef struct { uint32_t *data; int size; int cap; } U32VecI;

static void vinit(U32VecI *v){ v->data=NULL; v->size=0; v->cap=0; }
static void vfree(U32VecI *v){ free(v->data); v->data=NULL; v->size=0; v->cap=0; }

static void vreserve(U32VecI *v, int new_cap, MPI_Comm comm)
{
    if (new_cap <= v->cap) return;
    uint32_t *p = (uint32_t*)realloc(v->data, (size_t)new_cap * sizeof(uint32_t));
    if (!p) die_abort(comm, "exchangeplan: OOM (vreserve)");
    v->data = p; v->cap = new_cap;
}

static void vpush(U32VecI *v, uint32_t x, MPI_Comm comm)
{
    if (v->size == v->cap)
    {
        int nc = (v->cap > 0) ? 2*v->cap : 1024;
        vreserve(v, nc, comm);
    }
    v->data[v->size++] = x;
}

void exchangeplan_free(ExchangePlan *P)
{
    if (!P) return;

    free(P->sources);
    free(P->dests);

    free(P->need_from_counts);
    free(P->send_to_counts);
    free(P->need_from_displs);
    free(P->send_to_displs);

    free(P->need_from_flat);
    free(P->send_to_flat);
    free(P->need_from_gidx_flat);

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
    if (!P) return 1;
    exchangeplan_free(P);

    MPI_Comm_size(comm, &P->comm_size);
    MPI_Comm_rank(comm, &P->comm_rank);
    P->comm_graph = MPI_COMM_NULL;

    const int comm_size = P->comm_size;
    const int comm_rank = P->comm_rank;

    /* Build need_from[p] lists */
    U32VecI *need_from = (U32VecI*)malloc((size_t)comm_size * sizeof(U32VecI));
    if (!need_from) die_abort(comm, "exchangeplan_build: OOM need_from");
    for (int p=0; p<comm_size; ++p) vinit(&need_from[p]);

    for (uint32_t gi=0; gi<ghost_count; ++gi)
    {
        uint32_t v = ghost_vertices[gi];
        int owner = owner_fn(v, n_global, comm_size);
        if (owner != comm_rank)
            vpush(&need_from[owner], v, comm);
    }

    P->need_from_counts = (int*)malloc((size_t)comm_size * sizeof(int));
    P->send_to_counts   = (int*)malloc((size_t)comm_size * sizeof(int));
    P->need_from_displs = (int*)malloc((size_t)comm_size * sizeof(int));
    P->send_to_displs   = (int*)malloc((size_t)comm_size * sizeof(int));
    if (!P->need_from_counts || !P->send_to_counts || !P->need_from_displs || !P->send_to_displs)
        die_abort(comm, "exchangeplan_build: OOM counts/displs");

    for (int p=0; p<comm_size; ++p) P->need_from_counts[p] = need_from[p].size;

    MPI_Alltoall(P->need_from_counts, 1, MPI_INT,
                 P->send_to_counts,   1, MPI_INT, comm);

    int total_need = 0, total_send = 0;
    for (int p=0; p<comm_size; ++p) { P->need_from_displs[p] = total_need; total_need += P->need_from_counts[p]; }
    for (int p=0; p<comm_size; ++p) { P->send_to_displs[p]   = total_send; total_send += P->send_to_counts[p]; }

    P->total_need_from = total_need;
    P->total_send_to   = total_send;

    if (total_need > 0)
    {
        P->need_from_flat = (uint32_t*)malloc((size_t)total_need * sizeof(uint32_t));
        if (!P->need_from_flat) die_abort(comm, "exchangeplan_build: OOM need_from_flat");
        for (int p=0; p<comm_size; ++p)
        {
            int cnt = need_from[p].size;
            if (cnt > 0)
            {
                memcpy(P->need_from_flat + P->need_from_displs[p],
                       need_from[p].data, (size_t)cnt * sizeof(uint32_t));
            }
        }
    }

    if (total_send > 0)
    {
        P->send_to_flat = (uint32_t*)malloc((size_t)total_send * sizeof(uint32_t));
        if (!P->send_to_flat) die_abort(comm, "exchangeplan_build: OOM send_to_flat");
    }

    MPI_Alltoallv(P->need_from_flat, P->need_from_counts, P->need_from_displs, MPI_UINT32_T,
                  P->send_to_flat,   P->send_to_counts,   P->send_to_displs,   MPI_UINT32_T,
                  comm);

    /* active neighbor sets */
    P->indegree = 0; P->outdegree = 0;
    for (int p=0; p<comm_size; ++p)
    {
        if (P->need_from_counts[p] > 0) ++P->indegree;
        if (P->send_to_counts[p]   > 0) ++P->outdegree;
    }

    P->sources = (P->indegree  > 0) ? (int*)malloc((size_t)P->indegree  * sizeof(int)) : NULL;
    P->dests   = (P->outdegree > 0) ? (int*)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    if ((P->indegree > 0 && !P->sources) || (P->outdegree > 0 && !P->dests))
        die_abort(comm, "exchangeplan_build: OOM sources/dests");

    int ii=0, oo=0;
    for (int p=0; p<comm_size; ++p)
    {
        if (P->need_from_counts[p] > 0) P->sources[ii++] = p;
        if (P->send_to_counts[p]   > 0) P->dests[oo++]   = p;
    }

    if (P->indegree > 0 || P->outdegree > 0)
    {
        MPI_Dist_graph_create_adjacent(comm,
                                       P->indegree,  P->sources, MPI_UNWEIGHTED,
                                       P->outdegree, P->dests,   MPI_UNWEIGHTED,
                                       MPI_INFO_NULL, 0, &P->comm_graph);
    }

    /* neighbor layouts */
    P->sendcounts = (P->outdegree > 0) ? (int*)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    P->sdispls    = (P->outdegree > 0) ? (int*)malloc((size_t)P->outdegree * sizeof(int)) : NULL;
    P->recvcounts = (P->indegree  > 0) ? (int*)malloc((size_t)P->indegree  * sizeof(int)) : NULL;
    P->rdispls    = (P->indegree  > 0) ? (int*)malloc((size_t)P->indegree  * sizeof(int)) : NULL;

    if ((P->outdegree > 0 && (!P->sendcounts || !P->sdispls)) ||
        (P->indegree  > 0 && (!P->recvcounts || !P->rdispls)))
        die_abort(comm, "exchangeplan_build: OOM neighbor counts/displs");

    for (int k=0; k<P->outdegree; ++k)
    {
        int p = P->dests[k];
        P->sendcounts[k] = P->send_to_counts[p];
        P->sdispls[k]    = P->send_to_displs[p];
    }
    for (int k=0; k<P->indegree; ++k)
    {
        int p = P->sources[k];
        P->recvcounts[k] = P->need_from_counts[p];
        P->rdispls[k]    = P->need_from_displs[p];
    }

    if (P->total_send_to > 0)
    {
        P->send_labels_flat = (uint32_t*)malloc((size_t)P->total_send_to * sizeof(uint32_t));
        if (!P->send_labels_flat) die_abort(comm, "exchangeplan_build: OOM send_labels_flat");
    }
    if (P->total_need_from > 0)
    {
        P->recv_labels_flat = (uint32_t*)malloc((size_t)P->total_need_from * sizeof(uint32_t));
        if (!P->recv_labels_flat) die_abort(comm, "exchangeplan_build: OOM recv_labels_flat");
    }

    /* mapping need_from_flat -> ghost index */
    if (P->total_need_from > 0)
    {
        P->need_from_gidx_flat = (uint32_t*)malloc((size_t)P->total_need_from * sizeof(uint32_t));
        if (!P->need_from_gidx_flat) die_abort(comm, "exchangeplan_build: OOM need_from_gidx_flat");

        for (int i=0; i<P->total_need_from; ++i)
        {
            uint32_t v = P->need_from_flat[i];
            int found = 0;
            uint32_t idx = lower_bound_u32(ghost_vertices, ghost_count, v, &found);
            if (!found) die_abort(comm, "exchangeplan_build: need_from vertex not in ghost set");
            P->need_from_gidx_flat[i] = idx;
        }
    }

    for (int p=0; p<comm_size; ++p) vfree(&need_from[p]);
    free(need_from);

    return 0;
}

void exchangeplan_exchange(ExchangePlan *P, uint32_t *ghost_labels, MPI_Comm comm)
{
    (void)comm; /* comm currently unused; keep parameter for API stability */

    if (!P) return;

    if (P->comm_graph != MPI_COMM_NULL &&
        (P->total_send_to > 0 || P->total_need_from > 0))
    {
        MPI_Neighbor_alltoallv(P->send_labels_flat, P->sendcounts, P->sdispls, MPI_UINT32_T,
                               P->recv_labels_flat, P->recvcounts, P->rdispls, MPI_UINT32_T,
                               P->comm_graph);
    }

    if (P->total_need_from > 0)
    {
        for (int i = 0; i < P->total_need_from; ++i)
        {
            uint32_t gidx = P->need_from_gidx_flat[i];
            ghost_labels[gidx] = P->recv_labels_flat[i];
        }
    }
}
