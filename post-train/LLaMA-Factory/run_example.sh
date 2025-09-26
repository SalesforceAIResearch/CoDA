export HF_HOME=your-cache-dir
export WANDB_API_KEY=your-wandb-key

# cuda
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# FORCE_TORCHRUN=0 llamafactory-cli train examples/train_full/gpt2_preprocess.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/gpt2_full_ddm.yaml