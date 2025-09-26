# Overview
Training scripts for training large diffusion language models (e.g., Llama 7B).


## Installation
The code is tested on Python 3.10.12, with several 4xgh200 nodes on a slurm based cluster with The NVIDIA container image for PyTorch, release 24.07, available on [NGC](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-07.html). Since, the container includes several packages not listed in the requirements.txt, we include a pip-freeze of the package list for reference. We only implement the flash-attention version for LLama models. For code without flash-attention, please refer to the Llama factory diffusion adaption code in our repo.

```bash
pip install -r requirements.txt
```

## Data processing

We borrow data processing and dataloaders from TinyLlama. Please preprocess and tokenize the dataset following [them](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)

## Usage
For multi node runs, we prepare a list of node names in the cluster written in hostnames.txt that the base node can login into via ssh. 
```python
bash multi_node.sh
```
For single node runs, we do not need a list of node names. We directly use the accelerate command in. Note that the command uses the number of nodes to identify the world size, which can be set based on the machine. 
```python
bash run_distributed.sh
```





## Acknowledgements
This work is built on top of the following papers/repositories:
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention)
- [Yunchang](https://github.com/feifeibear/long-context-attention)
- [EasyContext](https://github.com/jzhang38/EasyContext/tree/main)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)


