# CoDA: Coding LM via Diffusion Adaptation

<p align="center">
  <img src="CoDA-logo.png" alt="CoDA Logo"/>
</p>

> End-to-end diffusion language modeling across TPU pre-training, GPU fine-tuning, evaluation, and serving.

[![Paper](https://img.shields.io/badge/Technical_Report-📄-lightgrey?logo=arxiv&logoColor=red)](technical_report.pdf) [![Model](https://img.shields.io/badge/Model-CoDA_Instruct-ff69b4?logo=huggingface&logoColor=yellow)](https://huggingface.co/Salesforce/CoDA-v0-Instruct)

---

## Table of Contents
- [Overview 🎯](#overview-)
- [Repository Map 🗺️](#repository-map-️)
- [Training Quickstart 🚀](#training-quickstart-)
  - [Pre-training on TPU (`pre-train/`)](#pre-training-on-tpu-pre-train)
  - [Supervised Diffusion Fine-tuning (`post-train/`)](#supervised-diffusion-fine-tuning-post-train)
  - [Evaluation Pipelines (`evaluation/`)](#evaluation-pipelines-evaluation)
- [Benchmark 📊](#benchmark-)
- [Deployment Guide 🛠️](#deployment-guide-)
- [Citation 📚](#citation-)

## Overview 🎯
CoDA is Salesforce AI Research's open diffusion language model. This repo contains a unified training pipeline from pre-training to post-training, evaluation harnesses, and a simple Fast-API based serving backend.
>Note: This repository is provided for research purposes only. Data release is subjected to internal regulations.

## Repository Map 🗺️
| Directory | Purpose |
| --- | --- |
| `CoDALanguageModel/` | Huggingface model class |
| `post-train/` | Supervised fine-tuning (SFT) pipeline  |
| `evaluation/` | Evaluation framework |
| `pre-train/` | TPU-based pre-training pipeline|
| `serving/` | Serving stack |
| `run_sft.sh` | Launcher coupling pre-training checkpoints with the post-training diffusion trainer. |
| `save_hf_model.py` | Util function to convert checkpoint in Huggingface model class|


## Training Quickstart 🚀
To avoid dependency conflicts, we recommend maintaining **isolated environments per subsystem** and activate the corresponding environment before executing scripts in each subdirectory.


### Pre-training on TPU (`pre-train/`)
1. Populate TPU metadata in `pre-train/env.example` and copy to `pre-train/.env`.
2. Run `pre-train/setup_tpu.sh` to provision dependencies and sync the repository to the TPU pod.
3. Launch pre-training with the provided recipes (e.g., `pre-train/recipes/midtrain_v4_512.sh`) to produce CoDA checkpoints (GCS or local storage).

### Supervised Diffusion Fine-tuning (`post-train/`)
1. Install prerequisites following `post-train/LLaMA-Factory/README.md`.
2. Configure dataset metadata in `post-train/LLaMA-Factory/data/dataset_info.json` and diffusion arguments in `post-train/LLaMA-Factory/examples/train_full/*.yaml`.
3. Execute `./run_sft.sh` to fine-tune CoDA checkpoints with discrete denoising objectives.

### Evaluation Pipelines (`evaluation/`)
1. Choose a benchmark script such as `evaluation/lm_eval/eval_mbpp_humaneval.sh`.
2. Update `MODEL_DIR` and diffusion parameters (`diffusion_steps`, `temperature`, `top_p`) to match the target checkpoint.
3. Run the script to gather metrics; logs are stored locally for aggregation and reporting.


## Benchmark 📊
Comparison of code-generation performance across standard and plus-enhanced benchmarks. Evalplus is computed as the mean pass@1 on enhanced variants. Bold marks results where CoDA produces the strongest diffusion-model performance.

| Model | Humaneval Instruct | Humaneval Plus | MBPP Instruct | MBPP Plus | Evalplus |
| --- | --- | --- | --- | --- | --- |
| CoDA-Base | 29.3 | 23.8 | 35.2 | 46.0 | 34.9 |
| CoDA-Instruct | 54.3 | 47.6 | 47.2 | **63.2** | **55.4** |
| Dream-Base | 56.7 | 50.0 | 68.7 | 57.4 | 53.7 |
| Dream-7B-Instruct | 57.9 | 53.7 | 68.3 | 56.1 | 54.9 |
| LLaDA-8B-Instruct | 35.4 | 31.7 | 31.5 | 28.6 | 30.2 |
| Qwen3-1.7B | 66.5 | 61.6 | 46.2 | 65.9 | 63.8 |
| Qwen2.5-Coder-1.5B | 43.9 | 36.6 | 69.2 | 58.6 | 47.6 |
| Qwen2.5-Coder-1.5B-Instruct | 70.7 | 66.5 | 69.2 | 59.4 | 62.3 |
| Gemma-3-1B-it | 39.6 | 35.4 | 39.4 | 63.5 | 49.5 |
| LLaMA-3.2-1B-Instruct | 35.4 | 31.1 | 24.4 | 53.7 | 42.4 |


## Deployment Guide 🛠️
### 1 Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2 Install dependencies
```bash
pip install -r serving/requirements.txt
```

### 3 Export your Hugging Face token
```bash
export HF_TOKEN="hf_..."
```

### 4 Serve the model on cuda
```bash
bash serving/fast-api/start_server.sh
```
The server will listen on http://localhost:8000.

### 5 Interact with the served model
```bash
python serving/fast-api/chat_cli.py --base-url http://localhost:8000 --model Salesforce/CoDA-v0-Instruct
```
Optional flags:
- `--stream` to stream tokens as they are generated
- `--show-meta` to display latency and token usage

### Generation hyperparameters (env vars)
You can customize generation with these environment variables (defaults in parentheses):
- `MAX_TOKENS` (256)
- `TEMPERATURE` (0.0)
- `TOP_P` (unset)
- `TOP_K` (unset)
- `STEPS` (128)
- `ALG` ("entropy")
- `ALG_TEMP` (0.1)
- `BLOCK_LENGTH` (32)

Example:
```bash
export MAX_TOKENS=512
export TEMPERATURE=0.7
export TOP_P=0.9
export STEPS=128
export ALG=entropy
export ALG_TEMP=0.1
export BLOCK_LENGTH=32
bash serving/fast-api/start_server.sh
```


## Citation 📚
```
@misc{coda2025,
  title={CoDA: Coding LM via Diffusion Adaptation},
  author={Chen, Haolin and Wang, Shiyu and Qin, Can and Pang, Bo and Liu, Zuxin and Qiu, Jielin and Zhang, Jianguo and Zhou, Yingbo and Chen, Zeyuan and Xu, Ran and Heinecke, Shelby and Savarese, Silvio and Xiong, Caiming and Wang, Huan and Yao, Weiran},
  year={2025},
  publisher={Salesforce AI Research}
}
```
