# pageindex-mvp

Minimal Streamlit app for document question-answering with VectifyAI/PageIndex.
Upload PDF or Markdown documents, index them with PageIndex, then ask questions
through the PageIndex agentic `Collection.query(...)` flow.

## Retrieval approach

The app now uses PageIndex directly and intentionally has no lexical fallback:

- indexes files with `PageIndexClient(...).collection(...).add(path)`;
- relies on PageIndex's hierarchical tree index instead of local chunks or embeddings;
- answers with `collection.query(...)`, so the PageIndex agent can call its tools
  (`list_documents`, `get_document`, `get_document_structure`, `get_page_content`)
  and retrieve tight page/line ranges;
- supports local self-hosted mode or PageIndex cloud mode when `PAGEINDEX_API_KEY`
  is configured.

If PageIndex is not installed/configured, indexing and chat fail explicitly instead
of falling back to a weaker lexical retriever.

## Requirements

- Linux/macOS shell with `bash`
- Python 3.10+
- `uv` (the installer bootstraps it with `python3 -m pip --user` if missing)
- PageIndex dependencies, including an LLM API key for local indexing/querying, or
  a PageIndex cloud API key

The install/run scripts are idempotent and use `~/venv/<basename project dir>` by default, for example `~/venv/pageindex-mvp`.

## Quick start

```bash
./install.sh
cp .env.example .env
source ./run.sh 0.0.0.0 8501
```

Then open `http://127.0.0.1:8501`.

## Configure PageIndex

For local self-hosted PageIndex, configure the LLM provider expected by your
PageIndex model. For OpenAI-compatible defaults, edit `.env`:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_AGENTS_DISABLE_TRACING=1
PAGEINDEX_MODEL=openai/gpt-4o-mini
PAGEINDEX_RETRIEVE_MODEL=openai/gpt-4o-mini
```

Pour un serveur OpenAI-compatible local ou privé, modifie `OPENAI_BASE_URL`, par exemple `http://localhost:8000/v1` pour vLLM ou `http://localhost:11434/v1` pour Ollama en mode OpenAI. L’interface Streamlit expose aussi ce champ dans la sidebar; au runtime l’app renseigne `OPENAI_BASE_URL` et `OPENAI_API_BASE` pour PageIndex/LiteLLM. Si LiteLLM affiche `Provider List`, ajoute le préfixe provider dans `PAGEINDEX_MODEL` et `PAGEINDEX_RETRIEVE_MODEL` (`openai/...`, `ollama_chat/...`, `vllm/...`, etc.).

`OPENAI_AGENTS_DISABLE_TRACING=1` est recommandé par défaut avec les serveurs locaux ou clés non-OpenAI : cela évite que l’OpenAI Agents SDK tente d’exporter des traces vers OpenAI et produise une erreur 401 non fatale.

For PageIndex cloud mode:

```bash
PAGEINDEX_API_KEY=your-pageindex-key
```

## H100 / DGX Spark notes

The Streamlit app is backend-agnostic; PageIndex/LiteLLM chooses the indexing and
retrieval model. On H100/DGX Spark hosts, run a GPU-serving backend such as
vLLM/Ollama if supported by your PageIndex/LiteLLM model string, then set the
corresponding model and provider environment variables. If you need to pin GPUs,
set `CUDA_VISIBLE_DEVICES` in `.env`.

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
