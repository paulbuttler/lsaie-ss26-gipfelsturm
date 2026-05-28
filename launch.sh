#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [gpus_per_node] [attn_backend] [mbs] [fp8] [recompute] [offload_layers] [environment] [compile]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 4b, 8b, 32b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
# GPUs/node: optional, default 4. Choices: 1, 2, 4. Use 1 for single-GPU baselines.
# Attn:      optional, default auto. Choices: auto, flash, fused, unfused, local
# MBS:       optional. Override the per-size micro-batch-size preset. Pass "" to skip when setting later args.
# FP8:       optional, default off. Choices: off, hybrid, e4m3
#              hybrid = e4m3 fwd (weights/activations), e5m2 bwd (gradients) — recommended for training
#              e4m3   = e4m3 everywhere (more aggressive, may reduce stability)
# Recompute: optional, default off. Choices: off, selective, full
#              selective = recompute only attention softmax (saves ~30% activation mem, minimal overhead)
#              full      = recompute all activations per layer (saves ~70% activation mem, ~10-15% throughput cost)
# Offload:   optional, default 0. CPU-offload activations of the first N layers.
# Env:       optional, default alps3. EDF container name under ~/.edf/. E.g. alps3, alps3-fa3.
# Compile:   optional, default off. Choices: off, on  (experimental; FP8+inductor has known issues)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8

#            ./launch.sh throughput 3b 50 1 1 auto 2                            # 3B single-GPU baseline
#            ./launch.sh throughput 3b 50 1 1 flash 2                           # FA2 backend
#            ./launch.sh throughput 3b 50 1 1 flash 2 off off 0 alps3-fa3       # FA3 via alps3-fa3 container
#            ./launch.sh throughput 3b 50 1 1 auto 4 off off 12                 # offload 12 layers to fit MBS=4

#            ./launch.sh throughput 8b 50 1 4 auto 2                            # 8B single-node baseline (DP=4)
#            ./launch.sh throughput 8b 50 1 4 auto 1 hybrid                     # FP8 at MBS=1 
#            ./launch.sh throughput 8b 50 1 4 auto 4 off off 16                 # offload 16 layers and push MBS to 4

#            ./launch.sh throughput 32b 50 2 4 auto                             # 32B 2-node TP=4 baseline (DP=2)
#            ./launch.sh throughput 32b 50 2 4 auto "" hybrid full              # FP8 + full recompute
#            ./launch.sh throughput 32b 50 2 4 auto "" hybrid off 32            # FP8 + offload 32 layers

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [gpus_per_node] [attn_backend] [mbs] [fp8] [recompute] [offload_layers] [environment]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [gpus_per_node] [attn_backend] [mbs] [fp8] [recompute] [offload_layers] [environment]}

GPUS_PER_NODE=${5:-4}
case $GPUS_PER_NODE in
    1|2|4) ;;
    *) echo "Invalid gpus_per_node: $GPUS_PER_NODE. Choose: 1, 2, 4"; exit 1 ;;
esac
NUMA_BIND=0-$((GPUS_PER_NODE - 1))  # GH200: GPU i ↔ NUMA i

ATTN_BACKEND=${6:-auto}
case $ATTN_BACKEND in
    auto|flash|fused|unfused|local) ;;
    *) echo "Unknown attention backend: $ATTN_BACKEND. Choose: auto, flash, fused, unfused, local"; exit 1 ;;
esac

MBS_OVERRIDE=${7:-}       # override per-size MBS preset, e.g., to push MBS=4 for 8B model (default: off)

FP8=${8:-off}
case $FP8 in
    off|hybrid|e4m3) ;;
    *) echo "Unknown fp8 mode: $FP8. Choose: off, hybrid, e4m3"; exit 1 ;;
esac

RECOMPUTE=${9:-off}
case $RECOMPUTE in
    off|selective|full) ;;
    *) echo "Unknown recompute mode: $RECOMPUTE. Choose: off, selective, full"; exit 1 ;;
esac
OFFLOAD_LAYERS=${10:-0}   # coarse CPU activation offload of first N layers; 0 = off (default), max = num_layers-1

ENVIRONMENT=${11:-alps3}

COMPILE=${12:-off}
case $COMPILE in
    off|on) ;;
    *) echo "Unknown compile mode: $COMPILE. Choose: off, on"; exit 1 ;;
esac

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:30:00
        [ "$GPUS_PER_NODE" = "1" ] && TIME=01:00:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA="
    --log-timers-to-tensorboard"
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

# Compile jobs need extra walltime for first-run Triton/inductor compilation
[ "$COMPILE" = "on" ] && TIME=02:00:00

################ Model config ################
TP=1
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    4b)
        NUM_LAYERS=32; HIDDEN=3456; FFN=9216;  HEADS=27; KV_HEADS=9
        MBS=2
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    32b)
        NUM_LAYERS=64; HIDDEN=5120; FFN=27648; HEADS=40; KV_HEADS=8
        MBS=1; TP=4
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 4b, 8b, 32b"
        exit 1
        ;;
esac

if [ "$TP" -gt "$GPUS_PER_NODE" ]; then
    echo "Error: $MODEL_SIZE requires TP=$TP but gpus_per_node=$GPUS_PER_NODE. Pass at least $TP GPUs/node."
    exit 1
fi

MBS=${MBS_OVERRIDE:-$MBS}   # GBS must stay divisible by MBS*DP
if [ "$OFFLOAD_LAYERS" -lt 0 ] || [ "$OFFLOAD_LAYERS" -ge "$NUM_LAYERS" ]; then
    echo "Invalid offload_layers: $OFFLOAD_LAYERS. Must be 0..$((NUM_LAYERS - 1)) for $MODEL_SIZE"; exit 1
fi

GBS=256
SEQ_LEN=4096
RECOMPUTE_SUFFIX=$( [ "$RECOMPUTE" != "off" ] && echo "-recompute${RECOMPUTE}" || echo "" )
COMPILE_SUFFIX=$( [ "$COMPILE" = "on" ] && echo "-compile" || echo "" )
TP_COMM_SUFFIX=$( [ "$TP" -gt 1 ] && echo "-tpcomm-seqpar" || echo "" )
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-${GPUS_PER_NODE}g-${ATTN_BACKEND}-mbs${MBS}-fp8${FP8}${RECOMPUTE_SUFFIX}${COMPILE_SUFFIX}${TP_COMM_SUFFIX}-off${OFFLOAD_LAYERS}-${ENVIRONMENT}"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Offload args ################
# Memory-saving but throughput-negative where the model already fits.
if [ "$OFFLOAD_LAYERS" -gt 0 ]; then
    OFFLOAD_ARGS="--cpu-offloading-num-layers $OFFLOAD_LAYERS"
else
    OFFLOAD_ARGS=""
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
#SBATCH --chdir=${WORKDIR}
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: $(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=\$WORKDIR/.cache/dataset
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}
NUMA_BIND=${NUMA_BIND}
OFFLOAD_ARGS="${OFFLOAD_ARGS}"

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-\${SLURM_NNODES}n-\${SLURM_GPUS_PER_NODE}g-${ATTN_BACKEND}-mbs${MBS}-fp8${FP8}${RECOMPUTE_SUFFIX}${COMPILE_SUFFIX}${TP_COMM_SUFFIX}-off${OFFLOAD_LAYERS}-${ENVIRONMENT}
LOG_DIR=\$WORKDIR/runs/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=$WORKDIR/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORKDIR/.cache/inductor
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export NVTE_DEBUG=${NVTE_DEBUG:-0}
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

SETUP

cat >> "$SCRIPT" << TE_ARGS

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --attention-backend ${ATTN_BACKEND}
TE_ARGS

if [ "$FP8" != "off" ]; then
cat >> "$SCRIPT" << FP8_ARGS
    --fp8-format ${FP8}
FP8_ARGS
fi

if [ "$COMPILE" = "on" ]; then
cat >> "$SCRIPT" << 'COMPILE_ARGS'
    --torch-compile
    --torch-compile-mode reduce-overhead
COMPILE_ARGS
fi

cat >> "$SCRIPT" << TE_ARGS_CLOSE
)
TE_ARGS_CLOSE

# torchrun env-var prefix. --environment=<edf> strips host env, so vars must ride on the
# torchrun command line. PYTORCH_CUDA_ALLOC_CONF=expandable_segments avoids FP8 optimizer-init OOM.
# UB_SKIPMC=1 disables CUDA Multicast (unsupported on Clariden RDMA), required for TP>1.
TORCHRUN_ENV_VARS=""
[ "$FP8" != "off" ] && TORCHRUN_ENV_VARS+="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
[ "$TP" -gt 1 ]    && TORCHRUN_ENV_VARS+="UB_SKIPMC=1 "
TORCHRUN_ENV_PREFIX=""
[ -n "$TORCHRUN_ENV_VARS" ] && TORCHRUN_ENV_PREFIX="env $TORCHRUN_ENV_VARS"

cat >> "$SCRIPT" << TORCHRUN_ENV_BLOCK
TORCHRUN_ENV_PREFIX="${TORCHRUN_ENV_PREFIX}"
TORCHRUN_ENV_BLOCK

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)
TRAINING

if [ "$RECOMPUTE" = "selective" ]; then
cat >> "$SCRIPT" << 'RECOMPUTE_SELECTIVE'
RECOMPUTE_ARGS=(
    --recompute-granularity selective
)
RECOMPUTE_SELECTIVE
elif [ "$RECOMPUTE" = "full" ]; then
cat >> "$SCRIPT" << 'RECOMPUTE_FULL'
RECOMPUTE_ARGS=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)
RECOMPUTE_FULL
else
cat >> "$SCRIPT" << 'RECOMPUTE_OFF'
RECOMPUTE_ARGS=()
RECOMPUTE_OFF
fi

cat >> "$SCRIPT" << REGULARIZATION
REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
REGULARIZATION

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

REST

cat >> "$SCRIPT" << DIST_ARGS
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    $( [ "${TP}" -gt 1 ] && echo "--sequence-parallel --tp-comm-overlap" )
)

DIST_ARGS

cat >> "$SCRIPT" << 'REST2'
LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST2

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="${TORCHRUN_ENV_PREFIX}torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    $OFFLOAD_ARGS \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << FOOTER

echo "CMD: \$TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=${ENVIRONMENT} --cpus-per-task \$SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=\$NUMA_BIND \$TRAINING_CMD"

echo "END TIME: \$(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "DRYRUN: skipping sbatch"
else
    sbatch "$SCRIPT"
fi
