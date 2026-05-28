#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=gipfel-test-flash-attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

mkdir -p logs

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK \
    bash -c 'python - <<PY
import importlib, torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))

import transformer_engine as te
import transformer_engine.pytorch as te_pt
print("transformer_engine:", te.__version__)

try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except ImportError as e:
    print("flash_attn: NOT INSTALLED -", e)

try:
    import flash_attn_interface as fa3
    print("flash_attn_interface (FA3):", getattr(fa3, "__version__", "available"))
except ImportError as e:
    print("flash_attn_interface (FA3): NOT INSTALLED -", e)

try:
    import cudnn
    print("cudnn (python):", cudnn.__version__ if hasattr(cudnn, "__version__") else "available")
except ImportError as e:
    print("cudnn (python frontend): NOT INSTALLED -", e)
print("torch cuDNN:", torch.backends.cudnn.version())

from transformer_engine.pytorch.attention import dot_product_attention as dpa_mod
print("TE attention module:", dpa_mod.__file__)
PY'