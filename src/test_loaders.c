#define _GNU_SOURCE
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "opt_parser.h"
#include "graph_dist.h"

// -------------------------------------------------------------
// Helper: detect file extension
// -------------------------------------------------------------
static int is_mtx(const char *path)
{
    const char *ext = strrchr(path, '.');
    if (!ext)
        return 0;
    return strcasecmp(ext, ".mtx") == 0 || strcasecmp(ext, ".txt") == 0;
}

static int is_mat(const char *path)
{
    const char *ext = strrchr(path, '.');
    if (!ext)
        return 0;
    return strcasecmp(ext, ".mat") == 0;
}

// -------------------------------------------------------------
// Measure load time of a file
// -------------------------------------------------------------
static void test_one_file(const char *path, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0)
    {
        printf("====================================================\n");
        printf("Testing file: %s\n", path);
        fflush(stdout);
    }

    // ---------------------------------------------------------
    // 1. Test loading as MTX (parallel)
    // ---------------------------------------------------------
    if (is_mtx(path))
    {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        DistCSRGraph Gd;
        int rc = load_dist_csr_from_file(path,
                                         /*symmetrize=*/1,
                                         /*drop_self_loops=*/1,
                                         &Gd,
                                         comm);

        double t1 = MPI_Wtime();

        if (rc != 0)
        {
            if (rank == 0)
            {
                fprintf(stderr, "Error: MTX loader returned rc=%d for %s\n", rc, path);
            }
        }
        else if (rank == 0)
        {
            printf("[MTX] Load time: %.6f seconds\n", t1 - t0);
            printf("[MTX] n=%d  m=%" PRId64 "\n", Gd.n_global, Gd.m_global);
        }

        // Free graph
        free_dist_csr(&Gd);
    }

    // ---------------------------------------------------------
    // 2. Test loading as MAT (rank-0 loader + scatter)
    // ---------------------------------------------------------
    if (is_mat(path))
    {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        DistCSRGraph Gd;
        int rc = load_dist_csr_from_file(path,
                                         /*symmetrize=*/1,
                                         /*drop_self_loops=*/1,
                                         &Gd,
                                         comm);

        double t1 = MPI_Wtime();

        if (rc != 0)
        {
            if (rank == 0)
            {
                fprintf(stderr, "Error: MAT loader returned rc=%d for %s\n", rc, path);
            }
        }
        else if (rank == 0)
        {
            printf("[MAT] Load time: %.6f seconds\n", t1 - t0);
            printf("[MAT] n=%d  m=%" PRId64 "\n", Gd.n_global, Gd.m_global);
        }

        // Free graph
        free_dist_csr(&Gd);
    }

    if (!is_mtx(path) && !is_mat(path) && rank == 0)
    {
        printf("Skipping unknown extension for %s\n", path);
    }
}

// -------------------------------------------------------------
// Main
// -------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Fix OpenMP thread count
    omp_set_num_threads(16);

    // ---------------------------------------------------------
    // Parse command-line options
    // Example: --files A.mtx,B.mtx,C.mat
    // ---------------------------------------------------------
    if (argc < 3 || strcmp(argv[1], "--files") != 0)
    {
        if (rank == 0)
        {
            fprintf(stderr,
                    "Usage: %s --files file1,file2,file3\n"
                    "   or:  %s --files 1:5\n"
                    "NOTE: You must modify test program if using numeric ranges to map to filenames.\n",
                    argv[0], argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *spec = argv[2];

    // Parse using your opt parser (string list)
    OptIntList dummy; // we don't actually want integers, but parser must run
    opt_int_list_init(&dummy);

    char *copy = strdup(spec);
    char *tok = strtok(copy, ",");
    int file_count = 0;

    // Count first
    while (tok)
    {
        ++file_count;
        tok = strtok(NULL, ",");
    }

    free(copy);

    // Extract file names
    char **files = malloc(file_count * sizeof(char *));
    copy = strdup(spec);
    tok = strtok(copy, ",");
    int idx = 0;
    while (tok)
    {
        files[idx++] = strdup(tok);
        tok = strtok(NULL, ",");
    }
    free(copy);

    // ---------------------------------------------------------
    // Run test for each file
    // ---------------------------------------------------------
    MPI_Barrier(comm);
    for (int i = 0; i < file_count; ++i)
    {
        MPI_Barrier(comm);
        test_one_file(files[i], comm);
        free(files[i]);
        MPI_Barrier(comm);
    }

    free(files);
    opt_int_list_free(&dummy);

    MPI_Finalize();
    return 0;
}
