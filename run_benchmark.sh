#!/bin/bash
#SBATCH --job-name=cc_mpi_benchmark
#SBATCH --output=cc_mpi_benchmark.out
#SBATCH --error=cc_mpi_benchmark.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1         # 1 MPI rank per node
#SBATCH --cpus-per-task=128         # 128 cores for OpenMP
#SBATCH --time=1:30:00
#SBATCH --partition=rome

module load gcc/13.2.0-iqpfkya  openmpi/5.0.3-rhzbeym matio

export GRAPH_NAMES=("com-Friendster" "mycielskian20")

# OpenMP setup
export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

cd ~/CC_mpi/

for GRAPH_NAME in "${GRAPH_NAMES[@]}"; do
    mpirun -np $SLURM_NTASKS ./bin/mpi_cc_benchmark -r 2 -c 8192 -e 1,2,4,8 -o results/${GRAPH_NAME}/ data/${GRAPH_NAME}.mat
done