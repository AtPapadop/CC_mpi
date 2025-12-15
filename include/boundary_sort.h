#ifndef BOUNDARY_SORT_H
#define BOUNDARY_SORT_H

#include <stdint.h>
#include "vec_helpers.h"

/* Helpers for sorting and deduplicating boundary edge lists. */
void boundary_edges_sort(BoundaryEdge *edges, uint64_t count);
uint64_t boundary_edges_dedup(BoundaryEdge *edges, uint64_t count);

#endif /* BOUNDARY_SORT_H */
