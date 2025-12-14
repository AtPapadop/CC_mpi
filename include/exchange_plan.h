#ifndef EXCHANGE_PLAN_H
#define EXCHANGE_PLAN_H

#include <stdint.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef int (*OwnerFn)(uint32_t v, uint32_t n_global, int comm_size);

  typedef struct
  {
    int comm_size;
    int comm_rank;

    int indegree;
    int outdegree;
    int *sources; /* indegree */
    int *dests;   /* outdegree */
    MPI_Comm comm_graph;

    int *need_from_counts; /* by rank */
    int *send_to_counts;   /* by rank */
    int *need_from_displs; /* by rank */
    int *send_to_displs;   /* by rank */

    uint32_t *need_from_flat;      /* vertices we need labels for (owned by others) */
    uint32_t *send_to_flat;        /* vertices others need labels for (owned by us) */
    uint32_t *need_from_gidx_flat; /* map need_from_flat[i] -> ghost index in ghost_vertices */

    int total_need_from;
    int total_send_to;

    /* neighbor-alltoallv arrays (indexed by neighbor order, not rank) */
    int *sendcounts;
    int *sdispls;
    int *recvcounts;
    int *rdispls;

    /* flat send/recv label buffers (match *_flat layouts) */
    uint32_t *send_labels_flat;
    uint32_t *recv_labels_flat;
  } ExchangePlan;

  /**
   * Build an exchange plan for the given sorted-unique ghost vertex list.
   *
   * - ghost_vertices must be sorted unique (ascending).
   * - ghost_count is its length.
   * - owner_fn must match the same vertex partition used by your DistCSRGraph loader.
   *
   * Returns 0 on success; aborts via MPI_Abort on OOM or internal errors.
   */
  int exchangeplan_build(ExchangePlan *P,
                         const uint32_t *ghost_vertices,
                         uint32_t ghost_count,
                         uint32_t n_global,
                         MPI_Comm comm,
                         OwnerFn owner_fn);

  /** Free all resources owned by the plan (safe to call on partially built plans). */
  void exchangeplan_free(ExchangePlan *P);

  /**
   * Perform the halo exchange using the plan:
   * - Caller must have filled P->send_labels, i.e. P->send_labels_flat[ ] for all send_to vertices.
   * - This function performs MPI_Neighbor_alltoallv and updates ghost_labels[ ] in-place
   *   using P->need_from_gidx_flat mapping.
   *
   * ghost_labels must be indexed by ghost index corresponding to ghost_vertices used in build().
   */
  void exchangeplan_exchange(ExchangePlan *P, uint32_t *ghost_labels, MPI_Comm comm);
  void exchangeplan_exchange_delta(const ExchangePlan *P,
                                        const uint32_t *comp_label,
                                        const uint32_t *comp_of,
                                        uint32_t v_start, uint32_t v_end,
                                        uint32_t *prev_sent,          /* length P->total_send_to */
                                        uint32_t *ghost_labels,       /* length ghost_count */
                                        MPI_Comm comm,
                                        uint64_t **sendbuf_io, int *sendcap_io,
                                        uint64_t **recvbuf_io, int *recvcap_io);
#ifdef __cplusplus
}
#endif

#endif /* EXCHANGE_PLAN_H */
