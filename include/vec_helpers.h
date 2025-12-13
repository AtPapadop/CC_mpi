#ifndef VEC_HELPERS_H
#define VEC_HELPERS_H

#include <stdint.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ---------- uint32_t vector with uint64 size (good for large collections) ---------- */
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

  /* ---------- uint32_t vector with int size (good for MPI counts/displs) ---------- */
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

  /* ---------- boundary edge/pair + their vectors (used by cc_mpi.c) ---------- */
  typedef struct
  {
    uint32_t rep;    /* local component representative (LOCAL id) */
    uint32_t remote; /* remote vertex (GLOBAL id) */
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

  typedef struct
  {
    uint32_t rep;  /* local component representative (LOCAL id) */
    uint32_t gidx; /* ghost index */
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

#ifdef __cplusplus
}
#endif

#endif /* VEC_HELPERS_H */
