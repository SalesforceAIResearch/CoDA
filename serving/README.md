# Fast-DLLM

Fast-DLLM is a diffusion-based Large Language Model (LLM) inference acceleration framework that supports efficient inference for models like Dream and LLaDA.

## Quick Start

### Install packages:
```bash
pip install -r Fast-dLLM/requirements.txt
```

### Serve the model

Export your token first:
```bash
export HF_TOKEN="hf_..."
```

Start the API server:
```bash
bash Fast-dLLM/fast-api/start_server.sh
```

The server listens on `http://localhost:8000`.

### Interact with the model

Use the provided CLI to chat with your served model:
```bash
python Fast-dLLM/fast-api/chat_cli.py --base-url http://localhost:8000 --model Salesforce/CoDA-v0-Instruct
```

Optional flags:
- `--stream` to stream tokens
- `--show-meta` to print latency and token usage

