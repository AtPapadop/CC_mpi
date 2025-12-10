#ifndef CC_MPI_H
#define CC_MPI_H

#include <stdint.h>
#include <mpi.h>
#include "graph_dist.h"

/**
 * Distributed connected components via label propagation (MPI + OpenMP, halo exchange).
 *
 * - Gd is a DistCSRGraph with rows partitioned across ranks.
 * - labels_global must have size Gd->n_global on ALL ranks.
 * - chunk_size: Used for OpenMP scheduling of local loops.
 *     1   => static scheduling
 *     > 0 => dynamic with given chunk
 *     <=0 => dynamic with DEFAULT_CHUNK_SIZE
 * - exchange_interval:
 *     >= 1, number of local iterations between halo exchanges.
 *     1 => exchange every iteration (baseline).
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
 * Count the number of connected components from the global labels array.
*/
uint32_t count_connected_components(const uint32_t *restrict labels_global, uint32_t n_global);
                                      

#endif /* CC_MPI_H */
