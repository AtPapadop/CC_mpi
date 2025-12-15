#define _POSIX_C_SOURCE 200112L

#include "vec_helpers.h"
#include "runtime_utils.h"

#include <stdlib.h>


void u32vec_init(U32Vec *v)
{
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void u32vec_free(U32Vec *v)
{
  free(v->data);
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void u32vec_reserve(U32Vec *v, uint64_t new_cap, MPI_Comm comm)
{
  if (new_cap <= v->cap)
    return;
  uint32_t *p = (uint32_t *)realloc(v->data, (size_t)new_cap * sizeof(uint32_t));
  if (!p)
    mpi_die_abort(comm, "OOM: u32vec_reserve");
  v->data = p;
  v->cap = new_cap;
}

void u32vec_push(U32Vec *v, uint32_t x, MPI_Comm comm)
{
  if (v->size == v->cap)
    u32vec_reserve(v, v->cap ? 2 * v->cap : 4096, comm);
  v->data[v->size++] = x;
}


void u32veci_init(U32VecI *v)
{
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void u32veci_free(U32VecI *v)
{
  free(v->data);
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void u32veci_reserve(U32VecI *v, int new_cap, MPI_Comm comm)
{
  if (new_cap <= v->cap)
    return;
  uint32_t *p = (uint32_t *)realloc(v->data, (size_t)new_cap * sizeof(uint32_t));
  if (!p)
    mpi_die_abort(comm, "OOM: u32veci_reserve");
  v->data = p;
  v->cap = new_cap;
}

void u32veci_push(U32VecI *v, uint32_t x, MPI_Comm comm)
{
  if (v->size == v->cap)
    u32veci_reserve(v, v->cap ? 2 * v->cap : 1024, comm);
  v->data[v->size++] = x;
}


void bevec_init(BEVec *v)
{
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void bevec_free(BEVec *v)
{
  free(v->data);
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void bevec_reserve(BEVec *v, uint64_t new_cap, MPI_Comm comm)
{
  if (new_cap <= v->cap)
    return;
  BoundaryEdge *p = (BoundaryEdge *)realloc(v->data, (size_t)new_cap * sizeof(BoundaryEdge));
  if (!p)
    mpi_die_abort(comm, "OOM: bevec_reserve");
  v->data = p;
  v->cap = new_cap;
}

void bevec_push(BEVec *v, BoundaryEdge e, MPI_Comm comm)
{
  if (v->size == v->cap)
    bevec_reserve(v, v->cap ? 2 * v->cap : 4096, comm);
  v->data[v->size++] = e;
}


void bpvec_init(BPVec *v)
{
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void bpvec_free(BPVec *v)
{
  free(v->data);
  v->data = NULL;
  v->size = 0;
  v->cap = 0;
}

void bpvec_reserve(BPVec *v, uint64_t new_cap, MPI_Comm comm)
{
  if (new_cap <= v->cap)
    return;
  BoundaryPair *p = (BoundaryPair *)realloc(v->data, (size_t)new_cap * sizeof(BoundaryPair));
  if (!p)
    mpi_die_abort(comm, "OOM: bpvec_reserve");
  v->data = p;
  v->cap = new_cap;
}

void bpvec_push(BPVec *v, BoundaryPair x, MPI_Comm comm)
{
  if (v->size == v->cap)
    bpvec_reserve(v, v->cap ? 2 * v->cap : 4096, comm);
  v->data[v->size++] = x;
}
