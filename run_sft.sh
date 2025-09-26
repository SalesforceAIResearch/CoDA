#!/bin/bash

# Simple TorchPrime DiffuLLaMA SFT Training Script

# Create logs directory if it doesn't exist
# mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="../../logs/torchprime_ddm_sft_${TIMESTAMP}.log"

# Navigate to the LLaMA-Factory directory
cd DiffuLLaMA/LLaMA-Factory

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1

# Add NCCL settings for better stability
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_TIMEOUT=1800
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 

echo "Starting TorchPrime DiffuLLaMA SFT training..." | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Run the training and save all output to log file with timestamps
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training command..."
    llamafactory-cli train examples/train_full/torchprime_ddm-sft-opc-stage1.yaml 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training command completed with exit code: $?"
} | tee -a "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "Training completed! Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE" 