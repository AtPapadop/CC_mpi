#ifndef GRAPH_DIST_H
#define GRAPH_DIST_H

#include <stdint.h>
#include "graph.h"
#include <mpi.h>

/**
 * Distributed Compressed Sparse Row (CSR) representation of a graph.
 * Each MPI process holds a partition of the graph.
 * - Global graph is partitioned by rows across ranks.
 * - Each rank owns a contiguous block of vertices [v_start, v_end).
 * - row_ptr is local (size n_local + 1), col_idx entries are global vertex IDs.
 */
typedef struct
{
  uint32_t n_global;
  uint64_t m_global;
  uint32_t n_local;
  uint64_t m_local;
  uint32_t v_start;
  uint32_t v_end;
  uint64_t *row_ptr;
  uint32_t *col_idx;
  uint32_t *part_bounds;
  int part_size;
  int part_kind;
} DistCSRGraph;

/**
 * Load an undirected graph from file into a distributed CSR representation.
 *
 *  - For .mtx/.txt: No longer supported
 *  - For .mat: Loads only from rank 0 and distributes to all ranks.
 * symmetrize = 1 → ensure undirected by adding reverse edges.
 * drop_self_loops = 1 → skip edges (i,i).
 * use_edge = 1 → use edge-balanced partitioning; 0 → vertex-balanced.
 * Returns 0 on success.
 */
int load_dist_csr_from_file(const char *path,
                            int symmetrize,
                            int drop_self_loops,
                            int use_edge,
                            DistCSRGraph *out,
                            MPI_Comm comm);
/**
 * Free memory allocated for a DistCSRGraph. (local arrays only)
 */
void free_dist_csr(DistCSRGraph *graph);

#endif // GRAPH_DIST_H