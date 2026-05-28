#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --job-name=fa3-build
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=01:30:00
#SBATCH --container-writable
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -eo pipefail
mkdir -p logs

echo "=== Build start: $(date) ==="
echo "Node: $(hostname)"

cd ~/fa3-build
podman build -t fa3-alps3:v3 .

echo "=== Build done: $(date) ==="
podman images | grep fa3-alps3

echo "=== Converting to sqsh: $(date) ==="
cd /iopsstor/scratch/cscs/$USER
enroot import -o fa3-alps3-v3.sqsh podman://fa3-alps3:v3

echo "=== sqsh done: $(date) ==="
ls -lh fa3-alps3-v3.sqsh
