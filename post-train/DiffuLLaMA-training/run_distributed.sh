export HF_TOKEN=<HF_TOKEN>
export HF_HOME=<DIRECTORY TO YOUR HUGGINGFACE HOME>
export HF_DATASETS_CACHE=<DIRECTORY TO YOUR CACHE FOLDER>

export TRITON_LIBCUDA_PATH=<CUDA DIR> e.g. /usr/local/cuda/compat/lib.real 
export CUDA_LAUNCH_BLOCKING=1

set -ex
export CUDA_DEVICE_MAX_CONNECTIONS=1
PBSNODEFILE=hostname.txt
export MASTER_ADDR=$(head -n 1 $PBSNODEFILE)

# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=5000
GPUS_PER_NODE=4
NNODES=`wc -l < $PBSNODEFILE`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODEID=$1 #RANDOM


config_json=accelerate_configs/single_node.yaml

export LAUNCHER="accelerate launch \
    --config_file $config_json \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODEID \
    --num_processes $WORLD_SIZE \
    --num_machines $NNODES \
    "

export CMD="train.py \
--batch-size 60 \
--gradient-accumulate-every 4  \
--output-dir ./output/7B_diffusion \
--seed 2829 \
--wandb Diffusion \
--max-train-steps 20000  \
--learning-rate 1.5e-5  \
--dataset /work/nvme/bcaq/slim_star_combined/ \
--model meta-llama/Llama-2-7b-hf  \
--seq-length 2048 \
--parallel_mode data_parallel \
"


$LAUNCHER $CMD