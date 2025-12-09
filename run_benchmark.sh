#!/bin/bash
#SBATCH --job-name=cc_mpi_benchmark
#SBATCH --output=cc_mpi_benchmark.out
#SBATCH --error=cc_mpi_benchmark.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:00:00
#SBATCH --partition=rome

module load gcc/9.4.0-eewq4j6  openmpi/5.0.3-6ysrffb matio

export GRAPH_NAME="com-Friendster"

cd ~/CC_mpi/
mpirun -np $SLURM_NTASKS bin/mpi_cc_benchmark -r 5 -c 1024,2048,4096,8192 -e 1,2,4,8 -f data/${GRAPH_NAME}.mat -o results/${GRAPH_NAME}/
