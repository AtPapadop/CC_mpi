#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <errno.h>
#include <matio.h>

#include "graph.h"
#include "mmio.h"

// simple (u,v) edge
typedef struct
{
  int32_t u, v;
} Edge;

// comparison function between edges for qsort
static int cmp_edge(const void *a, const void *b)
{
  const Edge *ea = (const Edge *)a;
  const Edge *eb = (const Edge *)b;
  if (ea->u != eb->u)
    return (ea->u < eb->u) ? -1 : 1;
  if (ea->v != eb->v)
    return (ea->v < eb->v) ? -1 : 1;
  return 0;
}

// deduplicate sorted edges in-place, updating m
static void dedup_edges(Edge *E, int64_t *m, int drop_self_loops)
{
  int64_t write = 0;
  for (int64_t r = 0; r < *m; ++r)
  {
    if (drop_self_loops && E[r].u == E[r].v)
      continue;
    if (write == 0 ||
        E[r].u != E[write - 1].u ||
        E[r].v != E[write - 1].v)
    {
      E[write++] = E[r];
    }
  }
  *m = write;
}

int load_csr_from_file(const char *path, int symmetrize, int drop_self_loops, CSRGraph *out)
{
  const char *ext = strrchr(path, '.');
  if (!ext)
    ext = "";

  if (strcasecmp(ext, ".mtx") == 0 || strcasecmp(ext, ".txt") == 0)
  {
    return load_csr_from_mtx(path, symmetrize, drop_self_loops, out);
  }
  else if (strcasecmp(ext, ".mat") == 0)
  {
    return load_csr_from_mat(path, out);
  }
  else
  {
    fprintf(stderr, "Unsupported file extension: %s\n", ext);
    return 99;
  }
}

int load_csr_from_mtx(const char *path, int symmetrize, int drop_self_loops, CSRGraph *out)
{
  memset(out, 0, sizeof(*out));
  FILE *f = fopen(path, "r");
  if (!f)
  {
    perror("fopen");
    return 1;
  }

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0)
  {
    fprintf(stderr, "Could not process Matrix Market banner.\n");
    fclose(f);
    return 2;
  }

  if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode))
  {
    fprintf(stderr, "Only sparse coordinate matrices are supported.\n");
    fclose(f);
    return 3;
  }

  int M, N, nz;
  if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
  {
    fprintf(stderr, "Failed reading size line.\n");
    fclose(f);
    return 4;
  }

  int32_t n = (int32_t)((M > N) ? M : N);
  int symmetric_in_file =
      mm_is_symmetric(matcode) || mm_is_hermitian(matcode) || mm_is_skew(matcode);

  // allocate edges (worst case doubling)
  int64_t cap = (int64_t)nz * (symmetric_in_file || symmetrize ? 2 : 1);
  Edge *E = (Edge *)malloc(sizeof(Edge) * cap);
  if (!E)
  {
    fclose(f);
    return 5;
  }

  int64_t m = 0;
  for (int k = 0; k < nz; ++k)
  {
    int i, j;
    if (mm_is_pattern(matcode))
    {
      if (fscanf(f, "%d %d", &i, &j) != 2)
        break;
    }
    else
    {
      double val;
      if (fscanf(f, "%d %d %lf", &i, &j, &val) < 2)
        break;
    }
    i--;
    j--; // convert to 0-based
    if (i < 0 || j < 0 || i >= n || j >= n)
      continue;
    E[m++] = (Edge){i, j};
    if ((symmetric_in_file || symmetrize) && i != j)
      E[m++] = (Edge){j, i};
  }
  fclose(f);

  qsort(E, m, sizeof(Edge), cmp_edge);
  dedup_edges(E, &m, drop_self_loops);

  int64_t *restrict row_ptr;
  if (posix_memalign((void **)&row_ptr, 64, sizeof(int64_t) * (n + 1)) != 0)
  {
    free(E);
    return 6;
  }

  for (int64_t r = 0; r < m; ++r)
    row_ptr[E[r].u + 1]++;
  for (int32_t i = 0; i < n; ++i)
    row_ptr[i + 1] += row_ptr[i];

  int32_t *restrict col_idx;
  if (posix_memalign((void **)&col_idx, 64, sizeof(int32_t) * m) != 0)
  {
    free(row_ptr);
    free(E);
    return 7;
  }

  int64_t *head = (int64_t *)malloc(sizeof(int64_t) * n);
  if (!head)
  {
    free(col_idx);
    free(row_ptr);
    free(E);
    return 8;
  }
  memcpy(head, row_ptr, sizeof(int64_t) * n);

  for (int64_t r = 0; r < m; ++r)
  {
    int32_t u = E[r].u;
    col_idx[head[u]++] = E[r].v;
  }

  free(head);
  free(E);

  out->n = n;
  out->m = m;
  out->row_ptr = row_ptr;
  out->col_idx = col_idx;
  return 0;
}

void free_csr(CSRGraph *g)
{
  if (!g)
    return;
  if (g->row_ptr)
    free(g->row_ptr);
  if (g->col_idx)
    free(g->col_idx);
  g->row_ptr = NULL;
  g->col_idx = NULL;
  g->n = 0;
  g->m = 0;
}

int load_csr_from_mat(const char *path, CSRGraph *out)
{
  memset(out, 0, sizeof(*out));

  mat_t *matfp = Mat_Open(path, MAT_ACC_RDONLY);
  if (!matfp)
  {
    fprintf(stderr, "Error opening .mat file: %s\n", path);
    return 1;
  }

  matvar_t *var = Mat_VarReadNext(matfp);
  while (var)
  {
    if (strcmp(var->name, "Problem") == 0 && var->data_type == MAT_T_STRUCT)
    {
      matvar_t *fieldA = Mat_VarGetStructFieldByName(var, "A", 0);
      if (fieldA)
        var = fieldA;
      break;
    }
    if (var->class_type == MAT_C_SPARSE)
      break;
    Mat_VarFree(var);
    var = Mat_VarReadNext(matfp);
  }

  if (!var)
  {
    fprintf(stderr, "No sparse matrix found in .mat file %s\n", path);
    Mat_Close(matfp);
    return 2;
  }

  mat_sparse_t *sparse = var->data;
  int32_t n = (int32_t)var->dims[0];
  int32_t mcols = (int32_t)var->dims[1];

  int64_t nz = sparse->jc[mcols];

  // Build CSR from column-compressed format in MATLAB (CSC)
  int64_t *row_ptr = (int64_t *)calloc((size_t)n + 1, sizeof(int64_t));
  int32_t *col_idx = (int32_t *)malloc(sizeof(int32_t) * nz);
  if (!row_ptr || !col_idx)
  {
    fprintf(stderr, "Memory allocation failed\n");
    Mat_VarFree(var);
    Mat_Close(matfp);
    free(row_ptr);
    free(col_idx);
    return 3;
  }

  // Count entries per row
  for (int c = 0; c < mcols; ++c)
  {
    for (int64_t j = sparse->jc[c]; j < sparse->jc[c + 1]; ++j)
    {
      int r = sparse->ir[j];
      if (r >= 0 && r < n)
        row_ptr[r + 1]++;
    }
  }
  for (int i = 0; i < n; ++i)
    row_ptr[i + 1] += row_ptr[i];

  // Fill col_idx
  int64_t *head = (int64_t *)malloc(sizeof(int64_t) * n);
  memcpy(head, row_ptr, sizeof(int64_t) * n);

  for (int c = 0; c < mcols; ++c)
  {
    for (int64_t j = sparse->jc[c]; j < sparse->jc[c + 1]; ++j)
    {
      int r = sparse->ir[j];
      if (r < 0 || r >= n)
        continue;
      col_idx[head[r]++] = c;
    }
  }

  free(head);
  Mat_VarFree(var);
  Mat_Close(matfp);

  out->n = n;
  out->m = row_ptr[n];
  out->row_ptr = row_ptr;
  out->col_idx = col_idx;
  return 0;
}
