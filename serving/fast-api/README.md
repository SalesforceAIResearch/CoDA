# Dream API — Quick Start

## Serve the model

Export your token first:
```bash
export HF_TOKEN="hf_..."
```

Start the server:
```bash
bash Fast-dLLM/fast-api/start_server.sh
```

The server will listen on `http://localhost:8000`.

## Interact with the model

Use the provided chat CLI:
```bash
python Fast-dLLM/fast-api/chat_cli.py --base-url http://localhost:8000 --model Salesforce/CoDA-v0-Instruct
```

Optional flags:
- `--stream` to stream tokens
- `--show-meta` to print latency and token usage