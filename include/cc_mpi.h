#ifndef CC_MPI_H
#define CC_MPI_H

#include <stdint.h>
#include <mpi.h>
#include "graph_dist.h"
#include "cc.h"

/**
 * Distributed connected components via:
 *   (1) Local CC per-rank using pthreads on the induced local subgraph (local-local edges only),
 *   (2) Iterative MPI halo exchange of boundary vertex labels to merge components across ranks.
 *
 * - Gd is a DistCSRGraph with rows partitioned across ranks.
 * - labels_global must have size Gd->n_global on ALL ranks.
 *
 * chunk_size:
 *   Passed to compute_connected_components_pthreads() as its chunk_size.
 *   1   => static scheduling in pthread code
 *   > 0 => dynamic with given chunk
 *   <=0 => dynamic with DEFAULT_CHUNK_SIZE (inside pthread code)
 *
 * exchange_interval
 *  NNumber of local iterations between successive MPI Halo exchanges.
 *
 * On return, labels_global[v] is the component representative (smallest vertex ID)
 * for vertex v, identical on all ranks.
 */
void compute_connected_components_mpi_advanced(const DistCSRGraph *restrict Gd,
                                               uint32_t *restrict labels_global,
                                               int chunk_size,
                                               int exchange_interval,
                                               MPI_Comm comm);

/**
 * Count the number of connected components from global labels array.
 * - labels_global: array of size n_global with component representatives.
 * - n_global: total number of vertices in the graph.
 * Returns the number of connected components.
 */
uint32_t count_connected_components(const uint32_t *restrict labels_global, uint32_t n_global);

/**
 * Set the number of threads to use for local CC computation per MPI rank.
 * If nthreads <= 0, the default number of threads is taken to be all available cores.
 */
void cc_mpi_set_num_threads(int nthreads);

#endif /* CC_MPI_H */
