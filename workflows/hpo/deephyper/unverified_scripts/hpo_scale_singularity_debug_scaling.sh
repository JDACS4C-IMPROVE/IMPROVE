#!/bin/bash
#PBS -l select=4:ncpus=64:ngpus=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -q debug-scaling
#PBS -A IMPROVE
#PBS -l filesystems=home:eagle
#PBS -N dh_hpo_scale_test
#PBS -M liu.yuanhang@mayo.edu

set -xe

# Move to the directory where `qsub example-improve.sh` was run
cd ${PBS_O_WORKDIR}

# source enviroemnt variabels for IMPROVE
source $IMPROVE_env

# Activate the current environement (module load, conda activate etc...)
module load PrgEnv-gnu
module use /soft/modulefiles 
module load conda
# activate base environment
conda_path=$(dirname $(dirname $(which conda)))
source $conda_path/bin/activate base
#source $conda_path/bin/activate $dh_env
# load module to run singularity container
module use /soft/spack/gcc/0.6.1/install/modulefiles/Core
module load apptainer

# Resource allocation for DeepHyper
export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE + 1))
export OMP_NUM_THREADS=$NDEPTH

echo NNODES: ${NNODES}
echo NTOTRANKS: ${NTOTRANKS}
echo OMP_NUM_THREADS: ${OMP_NUM_THREADS}

# GPU profiling, (quite ad-hoc, copy-paste the `profile_gpu_polaris.sh`, requires to install some small
# python package which queries nvidia-smi, you need a simple parser then to collect data.)
# UNCOMMENT IF USEFULL
# export GPUSTAT_LOG_DIR=$PBS_O_WORKDIR/$log_dir
# mpiexec -n ${NNODES} --ppn 1 --depth=1 --cpu-bind depth --envall ../profile_gpu_polaris.sh &

# Get list of process ids (basically node names)
echo $PBS_NODEFILE
export RANKS_HOSTS=$(python ./get_hosts_polaris.py $PBS_NODEFILE)

echo RANKS_HOSTS: ${RANKS_HOSTS}
echo PMI_LOCAL_RANK: ${PMI_LOCAL_RANK}

# Launch DeepHyper
# ensure that mpi is pointing to the one within deephyper conda environment
# set_affinity_gpu_polaris.sh does not seem to work right now
# but CUDA_VISIBLE_DEVICES was set within hpo_subprocess.py, 
mpiexec -n ${NTOTRANKS} -host ${RANKS_HOSTS} \
    --envall ./set_affinity_gpu_polaris.sh python hpo_subprocess_singularity.py