#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --job-name=gipfel-infra
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

WORKDIR=${WORKDIR:-$SLURM_SUBMIT_DIR}

mkdir -p logs

MASTER_ADDR=$(hostname)
MASTER_PORT=25678

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 \
    bash -c "torchrun \
        --nproc-per-node $SLURM_GPUS_PER_NODE \
        --nnodes $SLURM_NNODES \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --max_restarts 0 \
        --tee 3 \
        $WORKDIR/test_infra.py"
