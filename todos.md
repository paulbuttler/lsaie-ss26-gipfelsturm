# Onboarding
1. Make sure the infrastructure works by running `sbatch test-infra.sbatch`
2. Read the project README.md
3. Initialize the submodule by running `git submodule update --init --recursive`
4. Create `configs.sh` (gitignored) and set your W&B API Key.

# General
- initial setups
  - bash scripts for job submission
- further setups
  - profiling
  - checkpointing
- reproduce baselines

# Attention optimization

Replace the default attention backend with FlashAttention-
3, which exploits Hopper-specific hardware features
for faster attention computation. Potential extension:
Evaluate the cuDNN SDPA backend as an alternative.

# Low-precision training
Enable FP8 precision for matrix multiplications via
TransformerEngine, which can roughly double arith-
metic throughput on Hopper tensor cores while reduc-
ing activation memory. Potential extension: Evaluate
CPU activation offloading via the GH200’s NVLink-
C2C interconnect.

# Other

- alternative kernel fusions

# Report (deadline: May 29th)

Write a report
