# CC_mpi

## Build
1. Ensure an MPI toolchain and `libmatio` are available (e.g., via your module system).
2. From the repo root run `make`. Binaries land in `bin/` (notably `mpi_cc_benchmark`).

## Run `mpi_cc_benchmark`
Basic invocation:
```
mpirun -np <ranks> ./bin/mpi_cc_benchmark \
    -r <num_runs_per_config> \
    -c <chunk_size> \
    -e <comma_separated_edge_scale_factors> \
    -t <omp_threads_per_rank> \
    -o <output_dir> \
    <path_to_graph.(mtx|mat|txt)>
```
Typical example on a shared cluster node:
```
export OMP_NUM_THREADS=64
mpirun -np 4 ./bin/mpi_cc_benchmark -r 2 -c 8192 -e 1,2,4 -t ${OMP_NUM_THREADS} -o results/com-Friendster/ data/com-Friendster.mat
```

Flags:
- `-r` number of benchmark repetitions per configuration.
- `-c` chunk size for boundary exchanges.
- `-e` comma-separated list of sampling factors.
- `-t` OpenMP threads per MPI rank.
- `-o` directory where CSV results are written.

## SLURM script (`run_benchmark.sh`)
The provided script requests 4 nodes on the `rome` partition, launches one MPI rank per node (`--ntasks-per-node=1`) with 128 OpenMP threads each (`--cpus-per-task=128`), and submits a 30-minute job. It loads GCC, OpenMPI, and matio modules, sets `GRAPH_NAMES` (currently only `com-Friendster`), and for each graph runs `mpi_cc_benchmark` with the parameters shown in the script. Results are stored under `results/<graph_name>/`.
