#ifndef CC_H
#define CC_H

#include "graph.h"
#include <stdint.h>

// Default chunk size for parallel algorithms
#define DEFAULT_CHUNK_SIZE 4096

// Sequential connected components algorithm using label propagation
// Non optimal for sequential execution but simple to parallelize later
void compute_connected_components(const CSRGraph *restrict G, uint32_t *restrict labels);

// Connected components algorithm using BFS
// This is viable only for sequential execution
void compute_connected_components_bfs(const CSRGraph *restrict G, uint32_t *restrict labels);

// Parallel connected components algorithm using OpenMP
// Identical to the label propagation version but using OpenMP for loop parallelism
void compute_connected_components_omp(const CSRGraph *restrict G, uint32_t *restrict labels, int chunk_size);

// Parallel connected components algorithm using OpenCilk
// Identical to the label propagation version but using Cilk for loop parallelism
void compute_connected_components_cilk(const CSRGraph *restrict G, uint32_t *restrict labels, int chunk_size);

// Parallel connected components algorithm using pthreads
// More complex implementation using pthreads for parallelism
void compute_connected_components_pthreads(const CSRGraph *restrict G, uint32_t *restrict labels,
                                           int num_threads, int chunk_size);

// Count the number of unique labels in the labels array
// Using label propagation, we know that labels are in the range [0, n-1]
uint32_t count_unique_labels(const uint32_t *restrict labels, uint32_t n);

#endif
