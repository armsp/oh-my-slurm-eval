#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -A "****"
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_4090:1
#SBATCH -o logs/%j.log
#SBATCH -e errs/%j.err

# Load the required modules 
module load stack/2024-06 eth_proxy python/3.12.8 cuda/12.4.1 git-lfs
source /cluster/---/---/you/venv/bin/activate
export HF_HOME=/cluster/scratch/you/
export HUGGINGFACE_HUB_CACHE=/cluster/scratch/sharaj/
export HF_DATASETS_CACHE="/cluster/scratch/sharaj/datasets"
export TRITON_CACHE_DIR="/cluster/scratch/sharaj/triton"
export FLASHINFER_WORKSPACE_DIR="/cluster/scratch/sharaj/flash_infer_workspace"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
# FLASH_ATTN, FLASHINFER or XFORMERS

think=$1
size=${2:-full}  # default to full if not passed

# if think is passed run the thinking script
if [ "$think" == "think" ]; then
    echo "Running thinking script"
    python sft_think.py $size
else
    echo "Running generation script"
    python gen_eval_sft.py $size
fi