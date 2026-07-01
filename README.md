# llama.cpp-manager

A lightweight HTTP API tool for managing multiple [llama.cpp](https://github.com/ggml-org/llama.cpp) server instances. Provides a web UI for model management, instance lifecycle control, GPU monitoring, and multi-source model downloading.

## Features

- **Web UI** — Browser-based dashboard for all operations
- **Multi-instance** — Run multiple models simultaneously on one GPU/CPU
- **Model Download** — HuggingFace, ModelScope, HF-Mirror, direct URL
- **Real-time Monitoring** — GPU memory/utilization, CPU, RAM charts
- **Auto Restart** — Crash detection with configurable retry strategy
- **Config Persistence** — YAML config + preset templates
- **Health Checks** — Automatic readiness detection per instance

## Quick Start

### Prerequisites

- Python 3.10+
- llama.cpp compiled with CUDA support (`llama-server` binary)

### Install

```bash
git clone https://github.com/etworker/llama.cpp-manager.git
cd llama.cpp-manager
pip install -r requirements.txt
```

### Run

```bash
python main.py --port 9000
```

Open `http://localhost:9000` in your browser.

### Docker (optional)

```bash
docker build -t llama-cpp-manager .
docker run -p 9000:9000 -v /path/to/models:/models llama-cpp-manager
```

## Configuration

Edit `config.yaml` or use the web UI Config page:

```yaml
config:
  model_dir: /path/to/models          # Model storage directory
  llama_server_bin: /path/to/llama-server  # llama.cpp binary path
  port_range_start: 8080              # Instance port range
  port_range_end: 8180
  default_ctx_size: 4096
  default_host: "0.0.0.0"             # Bind address for instances
  max_instances: 4
  auto_restart: true
  auto_restart_max_retries: 3
```

### Presets

Presets are saved parameter templates for quick model startup:

```yaml
presets:
  qwen3-8b-chat:
    description: Qwen3 8B for chat
    ctx_size: 8192
    n_parallel: 2
    batch_size: 2048
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/system/info` | System info (GPU, CPU, RAM) |
| GET | `/api/system/health` | Health check |
| GET | `/api/models` | List local models |
| POST | `/api/models/download` | Download model |
| GET | `/api/instances` | List running instances |
| POST | `/api/instances` | Start new instance |
| DELETE | `/api/instances/{id}` | Stop instance |
| POST | `/api/instances/{id}/restart` | Restart instance |
| GET | `/api/presets` | List presets |
| POST | `/api/presets` | Create preset |

Full interactive API docs available at `/docs` (Swagger UI).

## Project Structure

```
llama.cpp-manager/
├── main.py          # FastAPI entry point
├── config.py        # YAML config management
├── models.py        # Pydantic data models
├── manager.py       # Instance lifecycle management
├── downloader.py    # Multi-source model downloader
├── monitor.py       # GPU/RAM/CPU monitoring
├── router.py        # API routes
├── config.yaml      # Default configuration
├── static/
│   └── index.html   # Web UI
└── requirements.txt
```

## Requirements

```
fastapi>=0.100.0
uvicorn>=0.23.0
httpx>=0.24.0
psutil>=5.9.0
pydantic>=2.0.0
pyyaml>=6.0
nvidia-ml-py>=12.0.0
huggingface-hub>=0.14.0
```

## License

MIT
