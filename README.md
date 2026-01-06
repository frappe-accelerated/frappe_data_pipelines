# Frappe Data Pipelines

Document embedding and vector search for Frappe Drive with Insights integration.

## Features

- Automatic text extraction from PDF, DOCX, TXT, MD files
- Chunking with LangChain text splitters
- Embedding via OpenRouter API or local sentence-transformers
- Vector storage in Qdrant (embedded or server mode)
- Row-Level Security (RLS) for document chunks
- Background job processing with retry and cleanup

## Installation

```bash
bench get-app https://github.com/frappe-accelerated/frappe_data_pipelines
bench --site your-site install-app frappe_data_pipelines
```

## Configuration

Go to **Data Pipeline Settings** to configure:
- Embedding provider (OpenRouter or local)
- Qdrant connection settings
- Chunk size and overlap
- Enabled file types

## License

MIT
