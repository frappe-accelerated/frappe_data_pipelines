# Frappe Data Pipelines

Document embedding and vector search for Frappe Drive with Insights integration.

## Features

- Automatic text extraction from PDF, DOCX, TXT, MD files
- Chunking with LangChain text splitters
- **Local embeddings** via Ollama (nomic-embed-text, mxbai-embed-large, etc.)
- **Cloud embeddings** via OpenRouter API (OpenAI, Cohere models)
- Vector storage in Qdrant (embedded or server mode)
- Row-Level Security (RLS) for document chunks
- Background job processing with retry and cleanup
- Auto-detected embedding dimensions based on model selection

## Installation

```bash
bench get-app https://github.com/frappe-accelerated/frappe_data_pipelines
bench --site your-site install-app frappe_data_pipelines
```

## Embedding Providers

### Local (Ollama) - Recommended for Privacy

Run embeddings locally using [Ollama](https://ollama.ai/). No data leaves your server.

**VRAM Requirements:**
| Model | Dimensions | VRAM | Notes |
|-------|-----------|------|-------|
| nomic-embed-text | 768 | ~275 MB | Recommended for most use cases |
| mxbai-embed-large | 1024 | ~670 MB | Higher quality embeddings |
| all-minilm | 384 | ~45 MB | Lightweight, fast |
| snowflake-arctic-embed | 1024 | ~670 MB | Good for retrieval |

```bash
# Install Ollama and pull a model
ollama pull nomic-embed-text
```

### OpenRouter - Cloud API

Use OpenAI or Cohere embedding models via OpenRouter. Requires API key.

**Supported Models:**
- openai/text-embedding-3-small (1536 dims)
- openai/text-embedding-3-large (3072 dims)
- cohere/embed-english-v3.0 (1024 dims)
- cohere/embed-multilingual-v3.0 (1024 dims)

## Configuration

Go to **Data Pipeline Settings** to configure:
- Embedding provider (Ollama or OpenRouter)
- Model selection (dimensions auto-detected)
- Qdrant connection settings
- Chunk size and overlap
- Enabled file types

## License

MIT
