#define _GNU_SOURCE
#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200112L

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "graph.h"
#include "cc.h"

// Thread arguments structure for pthreads
typedef struct
{
  const CSRGraph *G;          // Graph
  _Atomic uint32_t *labels;   // Atomic labels array
  atomic_int *changed;        // Atomic flag to indicate if any thread changed labels
  _Atomic uint64_t *next_vertex; // dynamic work index for chunk distribution
  uint32_t n;                 // Number of vertices
  int thread_id;              // Thread ID
  int num_threads;            // Total number of threads
  pthread_barrier_t *barrier; // Barrier for synchronization
  int chunk_size;             // Chunk size for work distribution
  int chunking_enabled;       // Flag to enable/disable chunking
  uint32_t block_start;       // Start index of the block for this thread
  uint32_t block_end;         // End index of the block for this thread
} ThreadArgs;

// Returns 1 when vertex u (or its neighbors) adopts a lower label
// Uses inline to avoid function call overhead in the inner loop
static inline int relax_vertex_label(uint32_t u, const uint64_t *restrict row_ptr,
                                     const uint32_t *restrict col_idx, _Atomic uint32_t *restrict labels)
{
  uint32_t old_label = atomic_load_explicit(&labels[u], memory_order_relaxed);
  uint32_t new_label = old_label;

  // Check neighbors for smaller labels
  for (uint64_t j = row_ptr[u]; j < row_ptr[u + 1]; j++)
  {
    uint32_t v = col_idx[j];
    uint32_t neighbor_label = atomic_load_explicit(&labels[v], memory_order_relaxed);
    if (neighbor_label < new_label)
      new_label = neighbor_label;
  }

  // Update label if a smaller one was found
  if (new_label < old_label)
  {
    uint32_t current = old_label;
    while (current > new_label &&
           !atomic_compare_exchange_weak_explicit(&labels[u], &current, new_label,
                                                  memory_order_relaxed, memory_order_relaxed))
    {
    }

    // Propagate the new label to neighbors to help convergence
    for (uint64_t j = row_ptr[u]; j < row_ptr[u + 1]; j++)
    {
      uint32_t v = col_idx[j];
      uint32_t neighbor = atomic_load_explicit(&labels[v], memory_order_relaxed);
      while (neighbor > new_label &&
             !atomic_compare_exchange_weak_explicit(&labels[v], &neighbor, new_label,
                                                    memory_order_relaxed, memory_order_relaxed))
      {
      }
    }

    return 1;
  }

  return 0;
}

// Worker thread: Semi asynchronous label propagation
static void *lp_worker_full_async(void *arg)
{
  ThreadArgs *args = (ThreadArgs *)arg;
  const CSRGraph *G = args->G;
  const uint64_t *restrict row_ptr = G->row_ptr;
  const uint32_t *restrict col_idx = G->col_idx;
  const uint32_t n = args->n;
  const uint32_t chunk = (uint32_t)args->chunk_size;
  const int chunking_enabled = args->chunking_enabled;
  const uint32_t block_start = args->block_start;
  const uint32_t block_end = args->block_end;
  const uint64_t n_extent = n;

  while (1)
  {
    int local_changed = 0;

    // Reset dynamic work queue at the start of the round
    if (chunking_enabled && args->thread_id == 0)
      atomic_store_explicit(args->next_vertex, 0, memory_order_relaxed);
    pthread_barrier_wait(args->barrier);

    // Check if chunking is enabled and choose between dynamic or static work distribution
    if (chunking_enabled)
    {
      // Dynamic work distribution via atomic index in chunk_size blocks
      while (1)
      {
        uint64_t start = atomic_fetch_add_explicit(args->next_vertex, (uint64_t)chunk, memory_order_relaxed);
        if (start >= n_extent)
          break;
        uint64_t end = start + (uint64_t)chunk;
        if (end > n_extent)
          end = n_extent;

        for (uint32_t u = (uint32_t)start; u < (uint32_t)end; u++)
          local_changed |= relax_vertex_label(u, row_ptr, col_idx, args->labels);
      }
    }
    else
    {
      // Static block assigned to this thread
      for (uint32_t u = block_start; u < block_end; u++)
        local_changed |= relax_vertex_label(u, row_ptr, col_idx, args->labels);
    }

    // Mark if this thread changed anything
    if (local_changed)
      atomic_store_explicit(args->changed, 1, memory_order_relaxed);

    // Synchronize all threads
    pthread_barrier_wait(args->barrier);

    // One thread checks for convergence
    if (args->thread_id == 0)
    {
      if (atomic_load_explicit(args->changed, memory_order_acquire) == 0)
      {
        atomic_store_explicit(args->changed, -1, memory_order_release); // signal done
      }
      else
      {
        atomic_store_explicit(args->changed, 0, memory_order_relaxed); // reset flag
      }
    }

    pthread_barrier_wait(args->barrier);

    // Stop condition: changed == -1 means no thread changed anything
    if (atomic_load_explicit(args->changed, memory_order_acquire) == -1)
      break;
  }

  return NULL;
}

void compute_connected_components_pthreads(const CSRGraph *restrict G, uint32_t *restrict labels,
                                           int num_threads, int chunk_size)
{
  const uint32_t n = G->n;
  const int chunking_enabled = (chunk_size != 1);
  const int effective_chunk = (chunk_size > 0) ? chunk_size : DEFAULT_CHUNK_SIZE;
  const uint32_t static_block = (!chunking_enabled && num_threads > 0)
                                    ? (uint32_t)(((uint64_t)n + (uint64_t)num_threads - 1) / (uint64_t)num_threads)
                                    : 0;

  _Atomic uint32_t *atomic_labels;
  if (posix_memalign((void **)&atomic_labels, 64, (size_t)n * sizeof(*atomic_labels)) != 0)
  {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  for (uint32_t i = 0; i < n; i++)
    atomic_init(&atomic_labels[i], i);

  atomic_int changed;
  atomic_init(&changed, 1);

  // Dynamic work index for chunk-based scheduling
  _Atomic uint64_t next_vertex;
  atomic_init(&next_vertex, 0);

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, num_threads);

  pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
  ThreadArgs *args = malloc(num_threads * sizeof(ThreadArgs));

  // Create worker threads
  for (int t = 0; t < num_threads; t++)
  {
    args[t].G = G;
    args[t].labels = atomic_labels;
    args[t].changed = &changed;
    args[t].next_vertex = &next_vertex;
    args[t].n = n;
    args[t].thread_id = t;
    args[t].num_threads = num_threads;
    args[t].barrier = &barrier;
    args[t].chunk_size = chunking_enabled ? effective_chunk : 0;
    args[t].chunking_enabled = chunking_enabled;
    if (!chunking_enabled)
    {
      uint64_t start = (uint64_t)t * static_block;
      uint64_t end = start + static_block;
      if (start > n)
        start = n;
      if (end > n)
        end = n;
      args[t].block_start = (uint32_t)start;
      args[t].block_end = (uint32_t)end;
    }
    else
    {
      args[t].block_start = 0;
      args[t].block_end = 0;
    }
    pthread_create(&threads[t], NULL, lp_worker_full_async, &args[t]);
  }

  // Wait for all threads to finish
  for (int t = 0; t < num_threads; t++)
    pthread_join(threads[t], NULL);

  // Copy results
  for (uint32_t i = 0; i < n; i++)
    labels[i] = atomic_load_explicit(&atomic_labels[i], memory_order_relaxed);

  pthread_barrier_destroy(&barrier);
  free(atomic_labels);
  free(threads);
  free(args);
}
