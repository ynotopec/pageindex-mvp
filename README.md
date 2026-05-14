# pageindex-mvp

Minimal Streamlit app for PDF question-answering. Upload one or more PDFs, build a PageIndex-inspired hierarchical document index, then ask questions using either a local Ollama model or the popular OpenAI-compatible Chat Completions API.

## Retrieval approach

The app now uses a lightweight PageIndex-inspired, vectorless RAG pipeline:

- extracts PDF text page-by-page with PyMuPDF, with a pypdf fallback;
- detects natural headings and numbered sections to build a table-of-contents-style tree;
- retrieves section nodes with weighted title/summary/text scoring instead of only fixed-size chunks;
- injects document, section and page references into the LLM prompt for better traceability.

This remains a local MVP, not the full VectifyAI/PageIndex implementation or cloud OCR pipeline.

## Requirements

- Linux/macOS shell with `bash`
- Python 3.10+
- `uv` (the installer bootstraps it with `python3 -m pip --user` if missing)
- One LLM backend:
  - Ollama, for local/tokenless inference; or
  - an OpenAI-compatible API token and base URL

The install/run scripts are idempotent and use `~/venv/<basename project dir>` by default, for example `~/venv/pageindex-mvp`.

## Quick start

```bash
./install.sh
cp .env.example .env
source ./run.sh 0.0.0.0 8501
```

Then open `http://127.0.0.1:8501`.

## Configure an API token

Edit `.env`:

```bash
LLM_PROVIDER=openai_compatible
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

For Ollama:

```bash
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss
```

## H100 / DGX Spark notes

The app is backend-agnostic. On H100/DGX Spark hosts, run a GPU-serving backend such as Ollama, vLLM, or another OpenAI-compatible server on the host, then point `.env` at it. If you need to pin GPUs, set `CUDA_VISIBLE_DEVICES` in `.env`.

## systemd example

```ini
[Unit]
Description=pageindex-mvp
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/pageindex-mvp
ExecStart=/usr/bin/bash -lc '/opt/pageindex-mvp/run.sh 0.0.0.0 8501'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
