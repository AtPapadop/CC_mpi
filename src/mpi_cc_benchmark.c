#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "graph_dist.h"
#include "cc_mpi.h"
#include "opt_parser.h"
#include "results_writer.h"
#include "runtime_utils.h"

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS] <matrix-file-path>\n\n"
            "Options:\n"
            "  -c, --chunk-size SPEC       Chunk size list/range\n"
            "  -e, --exchange SPEC         Exchange interval list/range\n"
            "  -r, --runs N                Runs per configuration (default 1)\n"
            "  -o, --output DIR            Output directory (default 'results')\n"
            "  -t, --threads N             Pthreads per rank (default: all cores)\n"
            "  -h, --help                  Show help message\n",
            prog);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const char *chunk_spec = "2048";
    const char *exchange_spec = "1";
    int runs = 1;
    const char *output_dir = "results";
    const char *matrix_path = NULL;
    int threads_opt = 0;

    const struct option long_opts[] = {
        {"chunk-size", required_argument, NULL, 'c'},
        {"exchange", required_argument, NULL, 'e'},
        {"runs", required_argument, NULL, 'r'},
        {"output", required_argument, NULL, 'o'},
        {"threads", required_argument, NULL, 't'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "c:e:r:o:t:h", long_opts, &idx)) != -1)
    {
        switch (opt)
        {
        case 'c':
            chunk_spec = optarg;
            break;
        case 'e':
            exchange_spec = optarg;
            break;
        case 'r':
            if (opt_parse_positive_int(optarg, &runs) != 0 || runs <= 0)
            {
                if (rank == 0)
                    fprintf(stderr, "Invalid runs: %s\n", optarg);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 't':
            if (opt_parse_positive_int(optarg, &threads_opt) != 0 || threads_opt <= 0)
            {
                if (rank == 0)
                    fprintf(stderr, "Invalid threads: %s\n", optarg);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            if (rank == 0)
                print_usage(argv[0]);
            MPI_Finalize();
            return EXIT_SUCCESS;
        default:
            if (rank == 0)
                print_usage(argv[0]);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (optind >= argc)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Missing matrix file path.\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    matrix_path = argv[optind];

    int threads_default = runtime_default_threads();
    int threads_to_use = (threads_opt > 0) ? threads_opt : threads_default;
    if (threads_to_use < 1)
        threads_to_use = 1;

    cc_mpi_set_num_threads(threads_to_use);

    if (rank == 0)
        printf("MPI CC Benchmark (MPI ranks=%d, pthreads/rank=%d)\n",
               size, threads_to_use);

    if (rank == 0 && results_writer_ensure_directory(output_dir) != 0)
    {
        fprintf(stderr, "Failed to create output directory '%s': %s\n",
                output_dir, strerror(errno));
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    OptIntList chunk_sizes;
    opt_int_list_init(&chunk_sizes);
    if (opt_parse_range_list(chunk_spec, &chunk_sizes, "chunk sizes") != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Invalid chunk-size specification.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    OptIntList exchange_intervals;
    opt_int_list_init(&exchange_intervals);
    if (opt_parse_range_list(exchange_spec, &exchange_intervals, "exchange intervals") != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Invalid exchange interval specification.\n");
        opt_int_list_free(&chunk_sizes);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    char results_path[PATH_MAX];
    if (rank == 0)
    {
        if (results_writer_build_results_path(results_path, sizeof(results_path),
                                              output_dir, "results_mpi",
                                              matrix_path) != 0)
        {
            fprintf(stderr, "Failed to build results path.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        printf("Saving results to: %s\n", results_path);
    }

    double *run_times = malloc((size_t)runs * sizeof(double));
    if (!run_times)
    {
        if (rank == 0)
            fprintf(stderr, "Memory allocation failed (run_times).\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Barrier(comm);
    DistCSRGraph Gd;
    double t_load_start = MPI_Wtime();
    int rc = load_dist_csr_from_file(matrix_path, 1, 1, 1, &Gd, comm);
    double t_load_end = MPI_Wtime();

    if (rc != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Error loading graph.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (rank == 0)
        printf("Graph loaded (n=%" PRIu32 ", m=%" PRIu64 ") in %.6f seconds.\n",
               Gd.n_global, Gd.m_global, t_load_end - t_load_start);

    for (size_t ci = 0; ci < chunk_sizes.size; ci++)
    {
        int chunk = chunk_sizes.values[ci];

        for (size_t ei = 0; ei < exchange_intervals.size; ei++)
        {
            int exchange = exchange_intervals.values[ei];

            if (rank == 0)
                printf("\n=== Testing chunk=%d, exchange=%d (%d run%s) ===\n",
                       chunk, exchange, runs, runs == 1 ? "" : "s");

            for (int r = 0; r < runs; r++)
            {
                MPI_Barrier(comm);
                double t0 = MPI_Wtime();

                compute_connected_components_mpi_advanced(&Gd, NULL, chunk, exchange, comm);

                double t1 = MPI_Wtime();
                run_times[r] = t1 - t0;

                if (rank == 0)
                {
                    printf("Run %d: %.6f seconds\n", r + 1, run_times[r]);
                    printf("(component count printed by CC via MPI_Reduce)\n");
                }
            }

            if (rank == 0)
            {
                char colname[128];
                snprintf(colname, sizeof(colname),
                         "chunk_%d_exchange_%d", chunk, exchange);

                results_writer_status st =
                    append_times_column(results_path, colname,
                                        run_times, (size_t)runs);

                if (st != RESULTS_WRITER_OK)
                    fprintf(stderr,
                            "Warning: CSV update failed (%d) for %s\n",
                            st, colname);

                printf("Saved column '%s' to %s\n", colname, results_path);
            }
        }
    }

    free_dist_csr(&Gd);
    free(run_times);
    opt_int_list_free(&chunk_sizes);
    opt_int_list_free(&exchange_intervals);

    if (rank == 0)
        printf("\nAll tests completed.\n");

    MPI_Finalize();
    return EXIT_SUCCESS;
}
