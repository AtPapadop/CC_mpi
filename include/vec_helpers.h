#ifndef VEC_HELPERS_H
#define VEC_HELPERS_H

#include <stdint.h>
#include <mpi.h>

/** Growable uint32_t vector tracked with 64-bit size. */
typedef struct
{
  uint32_t *data;
  uint64_t size;
  uint64_t cap;
} U32Vec;

void u32vec_init(U32Vec *v);
void u32vec_free(U32Vec *v);
void u32vec_reserve(U32Vec *v, uint64_t new_cap, MPI_Comm comm);
void u32vec_push(U32Vec *v, uint32_t x, MPI_Comm comm);

/** Growable uint32_t vector tracked with int size. */
typedef struct
{
  uint32_t *data;
  int size;
  int cap;
} U32VecI;

void u32veci_init(U32VecI *v);
void u32veci_free(U32VecI *v);
void u32veci_reserve(U32VecI *v, int new_cap, MPI_Comm comm);
void u32veci_push(U32VecI *v, uint32_t x, MPI_Comm comm);

/** Local representative and remote vertex describing a boundary edge. */
typedef struct
{
  uint32_t rep;
  uint32_t remote;
} BoundaryEdge;

typedef struct
{
  BoundaryEdge *data;
  uint64_t size;
  uint64_t cap;
} BEVec;

void bevec_init(BEVec *v);
void bevec_free(BEVec *v);
void bevec_reserve(BEVec *v, uint64_t new_cap, MPI_Comm comm);
void bevec_push(BEVec *v, BoundaryEdge e, MPI_Comm comm);

/** Local representative and ghost index pair. */
typedef struct
{
  uint32_t rep;
  uint32_t gidx;
} BoundaryPair;

typedef struct
{
  BoundaryPair *data;
  uint64_t size;
  uint64_t cap;
} BPVec;

void bpvec_init(BPVec *v);
void bpvec_free(BPVec *v);
void bpvec_reserve(BPVec *v, uint64_t new_cap, MPI_Comm comm);
void bpvec_push(BPVec *v, BoundaryPair x, MPI_Comm comm);

#endif /* VEC_HELPERS_H */
