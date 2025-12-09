#ifndef GRAPH_H
#define GRAPH_H

#include <stddef.h>
#include <stdint.h>

// Compressed Sparse Row (CSR) representation of a graph.
typedef struct
{
  int32_t n;        // number of vertices
  int64_t m;        // number of edges
  int64_t *row_ptr; // row pointers
  int32_t *col_idx; // column indices
} CSRGraph;

// Load an undirected graph from Matrix Market file into CSR form.
// symmetrize = 1 → ensure undirected by adding reverse edges.
// drop_self_loops = 1 → skip edges (i,i).
// Returns 0 on success.
int load_csr_from_mtx(const char *path, int symmetrize, int drop_self_loops, CSRGraph *out);

// Load an undirected graph from MATLAB .mat file into CSR form.
// Returns 0 on success.
int load_csr_from_mat(const char *path, CSRGraph *out);

// Load an undirected graph from file (either .mtx/.txt or .mat) into CSR form.
// symmetrize = 1 → ensure undirected by adding reverse edges.
// drop_self_loops = 1 → skip edges (i,i).
// Returns 0 on success.
int load_csr_from_file(const char *path, int symmetrize, int drop_self_loops, CSRGraph *out);

// Free memory allocated for CSR graph.
void free_csr(CSRGraph *g);

#endif
