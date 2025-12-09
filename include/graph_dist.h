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
  int32_t n_global; // total number of vertices in the global graph
  int64_t m_global; // total number of edges in the global graph
  int32_t n_local;  // number of vertices owned by this rank
  int32_t m_local;  // number of edges owned by this rank
  int32_t v_start;  // global vertex ID of the first vertex owned by this rank
  int32_t v_end;    // global vertex ID of the last vertex owned by this rank (exclusive)
  int64_t *row_ptr; // local row pointers (size n_local + 1)
  int32_t *col_idx; // column indices (global vertex IDs)
} DistCSRGraph;

/**
 * Load an undirected graph from file into a distributed CSR representation.
 *
 *  - For .mtx/.txt: all ranks open the file and each rank builds ONLY its own rows.
 *  - For .mat: rank 0 uses load_csr_from_file() and scatters CSR rows.
 *
 * symmetrize = 1 → ensure undirected by adding reverse edges.
 * drop_self_loops = 1 → skip (i,i).
 *
 * Returns 0 on success, same code on all ranks.
 */
int load_dist_csr_from_file(const char *path,
                            int symmetrize,
                            int drop_self_loops,
                            DistCSRGraph *out,
                            MPI_Comm comm);

                     

/**
 * Free memory allocated for a DistCSRGraph. (local arrays only)
 */
void free_dist_csr(DistCSRGraph *graph);

#endif // GRAPH_DIST_H