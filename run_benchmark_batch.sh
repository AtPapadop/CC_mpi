#!/bin/bash
#SBATCH --job-name=cc_mpi_benchmark_batch
#SBATCH --output=cc_mpi_benchmark_batch.out
#SBATCH --error=cc_mpi_benchmark_batch.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1      # 1 MPI process per node
#SBATCH --cpus-per-task=20       # 20 OpenMP threads per process
#SBATCH --time=01:00:00
#SBATCH --partition=batch

gcc/13.2.0-iqpfkya  openmpi/5.0.3-rhzbeym matio

export GRAPH_NAME="com-Friendster"

export OMP_NUM_THREADS=20
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

cd ~/CC_mpi/

mpirun -np $SLURM_NTASKS \
    bin/mpi_cc_benchmark \
    -r 5 \
    -c 1024,2048,4096,8192 \
    -e 1,2,4,8 \
    -o results/${GRAPH_NAME}_batch/
    data/${GRAPH_NAME}.mat
