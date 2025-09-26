#!/bin/bash

TPU_VM_NAME="<TPU_VM_NAME>" # Please use a single-host TPU (v4-8/v5-8)
TPU_ZONE="<TPU_ZONE>"
PROJECT="<PROJECT>"
BRANCH="<BRANCH>"
RECIPE="recipes/ckpt_consolidation.sh"

# Define checkpoint directories with their corresponding model configs and resume checkpoints
GCS_PREFIX="gs://<GCS_PREFIX>/"

# Associative arrays mapping checkpoint directories to their model configs and resume checkpoints
declare -A MODEL_CONFIG
declare -A CHECKPOINT_CONFIG

# Configure each checkpoint directory with its model and resume checkpoints
# Format: MODEL_CONFIG["checkpoint_dir"]="model_name"
#         CHECKPOINT_CONFIG["checkpoint_dir"]="[step1 step2 step3]"
MODEL_CONFIG["checkpoint_dir"]="coda-qwen-1b-tpu-v5p"
CHECKPOINT_CONFIG["checkpoint_dir"]="[]"

for checkpoint_dir in "${!CHECKPOINT_CONFIG[@]}"; do
    model="${MODEL_CONFIG[$checkpoint_dir]}"
    resume_checkpoints="${CHECKPOINT_CONFIG[$checkpoint_dir]}"
    checkpoint="${GCS_PREFIX}${checkpoint_dir}"
    echo "=================================================="
    echo "Running checkpoint consolidation:"
    echo "  Checkpoint dir: $checkpoint"
    echo "  Model config: $model"
    echo "  Resume checkpoint: $resume_checkpoints"
    echo "=================================================="
    # Run the gcloud command and wait for it to complete
    gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
        --zone=$TPU_ZONE \
        --project=$PROJECT \
        --tunnel-through-iap \
        --worker=all \
        --command='
        cd torchprime; \
        git fetch; \
        git checkout '"$BRANCH"'; \
        git pull; \
        source venv/bin/activate; \
        bash '"$RECIPE"' '"$model"' '"$checkpoint"' "'"$resume_checkpoints"'"'
        
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $checkpoint with model $model and checkpoint $resume_checkpoints"
    else
        echo "❌ Failed: $checkpoint with model $model and checkpoint $resume_checkpoints"
        echo "Do you want to continue with the next checkpoint? (y/n)"
        read -r response
        if [[ "$response" != "y" && "$response" != "Y" ]]; then
            echo "Stopping execution."
            exit 1
        fi
    fi
    
    echo ""
    echo "Waiting 10 seconds before next combination..."
    sleep 10
done

echo "🎉 All checkpoint consolidation jobs completed!"
