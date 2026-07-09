# Veria - RAG Document Query System

A production-ready Retrieval-Augmented Generation (RAG) system that enables intelligent querying of PDF, DOCX, and Markdown documents using local LLMs and vector embeddings. Veria provides a FastAPI-based REST API for document ingestion and natural language querying.

## Project Summary

Veria is an intelligent document processing and retrieval system that:

- Ingests PDF, DOCX, and Markdown documents through upload APIs
- Parses binary files using MarkItDown for accurate text extraction
- Generates embeddings using local Ollama models
- Stores embeddings in Elasticsearch for efficient similarity search
- Enables natural language querying with context-aware, streaming responses

The system is designed for privacy-conscious applications where documents and queries remain local, leveraging Ollama for local inference and MinIO for scalable document storage.

## Features

- **PDF, DOCX, and Markdown Ingestion**: Upload and store documents via REST API
- **Intelligent Parsing**: Convert PDFs and DOCX files to markdown using MarkItDown
- **Local Embeddings**: Generate vector embeddings with Ollama
- **Vector Search**: Elasticsearch-powered semantic search
- **Streaming Responses**: Server-Sent Events (SSE) from the prompt endpoint
- **Document Metadata**: Stores filename, author, category, and parsing timestamps
- **Chunking Strategy**: Configurable chunking with overlap for better retrieval quality
- **Comprehensive Logging**: Detailed runtime logging with Loguru
- **Health Checks**: Built-in `/health` endpoint for monitoring

## Architecture

```
┌─────────────┐
│   FastAPI   │
│   Endpoint  │
└──────┬──────┘
       │
       ├─────────────────────────────────────────────────────┐
       │                                                     │
       ▼                                                     ▼
┌──────────────────┐                              ┌──────────────────┐
│ DocumentService  │                              │  MinIO Storage   │
│                  │                              │                  │
│ - PDF / DOCX     │◄────────────────────────────│ - File Storage   │
│   Parsing         │                              │ - Object Mgmt    │
│ - Embedding      │                              └──────────────────┘
│ - Query Pipeline │
└────────┬─────────┘
         │
         ├─────────────────────────────────────────────────────┐
         │                                                     │
         ▼                                                     ▼
┌──────────────────┐                              ┌──────────────────┐
│ LlamaIndex       │                              │ Elasticsearch    │
│ Pipeline         │                              │                  │
│                  │                              │ - Vector Store   │
│ - Markdown Parser│─────────────────────────────►│ - Semantic Search│
│ - Chunking       │                              │ - Index Mgmt     │
│ - Embedding      │                              └──────────────────┘
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Ollama       │
│                  │
│ - Embeddings     │
│ - LLM Inference  │
└──────────────────┘
```

## Prerequisites

### Required Software

- **Python**: 3.11 or 3.12
- **Poetry**: For dependency management
- **MinIO**: S3-compatible object storage (local or remote)
- **Elasticsearch**: Vector database for embeddings storage
- **Ollama**: Local LLM and embedding model server

### Ollama Models

The repository is configured to use these Ollama models:

- Embedding model: `all-minilm:latest`
- LLM model: `llama3.2:3b`

```bash
ollama pull all-minilm:latest
ollama pull llama3.2:3b
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Veria
```

### 2. Install Dependencies

Using Poetry:

```bash
poetry install
```

Or using pip:

```bash
pip install -e .
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and populate it with your runtime settings.

Example values:

```env
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=ROOTUSER
MINIO_SECRET_KEY=kP3*qV9@jB6
MINIO_BUCKET_NAME=voice-agent-veria-context-documents-pdf
MINIO_BUCKET_NAME_MD=voice-agent-veria-context-documents-markdown
MINIO_BUCKET_NAME_DOCX=voice-agent-veria-context-documents-docx
MINIO_BUCKET_NAME_DOCX_MD=voice-agent-veria-context-documents-docx-md
MINIO_BUCKET_NAME_PDF_MD=voice-agent-veria-context-documents-pdf-md
ELASTIC_USERNAME=elastic
ELASTIC_PASSWORD=xL5Rh2495352wx2lhlCyXS9C
ELASTIC_URL=http://localhost:9200
ELASTIC_IGNORE_SSL_ERRORS=true
ES_INDEX_NAME=veria-agent-index
ES_CLUSTER_NAME=veria-agent-cluster
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=all-minilm:latest
```

### 4. Start Required Services

**MinIO** (local):

```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e MINIO_ROOT_USER=ROOTUSER \
  -e MINIO_ROOT_PASSWORD=MINIOROOTPASSWORD \
  minio/minio server /data --console-address ":9001"
```

**Elasticsearch** (local):

```bash
docker run -d \
  -p 9200:9200 \
  -p 9300:9300 \
  --name elasticsearch \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

**Ollama**:

```bash
# Install Ollama from https://ollama.ai
ollama serve
```

## Configuration

### Environment Configuration

Runtime settings are loaded from `.env` by `src/Infrastructure/dbcontext/context.py`.

Required values include MinIO buckets, Elasticsearch connection settings, and Ollama model endpoints.

### MinIO Buckets

The application creates and uses these buckets on startup:

- `MINIO_BUCKET_NAME`
- `MINIO_BUCKET_NAME_MD`
- `MINIO_BUCKET_NAME_DOCX`
- `MINIO_BUCKET_NAME_DOCX_MD`
- `MINIO_BUCKET_NAME_PDF_MD`

Uploaded PDF and DOCX files are converted to markdown and saved into the corresponding markdown buckets.

### Embeddings and LLM

- Embeddings are generated in `src/Infrastructure/embedding_model/index.py` using `OllamaEmbedding`.
- LLM inference is configured in `src/Infrastructure/llm/index.py` using `Ollama` and a streaming response synthesizer.
- Elasticsearch vector store initialization is handled in `src/Infrastructure/elastic_search/index.py`.

## Usage

### Starting the Server

```bash
uvicorn src.main:app --port 3002 --reload --loop asyncio
```

The API will be available at `http://localhost:3002`.

### API Endpoints

#### Health Check

```bash
curl http://localhost:3002/health
```

Response:

```json
{ "status": "ok" }
```

#### Upload Markdown Document

```bash
curl -X POST http://localhost:3002/upload-md \
  -F "file=@/path/to/document.md"
```

#### Upload PDF Document

```bash
curl -X POST http://localhost:3002/upload-pdf \
  -F "file=@/path/to/document.pdf"
```

#### Upload DOCX Document

```bash
curl -X POST http://localhost:3002/upload-docx \
  -F "file=@/path/to/document.docx"
```

#### Query Documents

```bash
curl -X POST http://localhost:3002/prompt \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key points about financial regulations?"}'
```

The `/prompt` endpoint returns a streaming response via SSE.

### Python Client Example

```python
import requests

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:3002/upload-pdf",
        files={"file": f}
    )
    print(response.json())

# Query the documents
response = requests.post(
    "http://localhost:3002/prompt",
    json={"query": "Summarize the document"}
)

for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
```

## Request Flow

### Upload Flow

1. **File Upload** (`POST /upload-pdf`, `/upload-docx`, `/upload-md`)
   - Validates file type
   - Stores raw file in MinIO via `DocumentService.store_to_minio()`

2. **Document Parsing** (`DocumentService.parse_document()`)
   - Retrieves the file from MinIO
   - Converts PDF or DOCX to markdown using MarkItDown
   - Builds a LlamaIndex document with metadata

3. **Embedding Generation**
   - Runs the ingestion pipeline: Markdown parser → Sentence splitter → Ollama embedding model
   - Stores embeddings in Elasticsearch

### Query Flow

1. **Query Submission** (`POST /prompt`)
   - Receives natural language query
   - Retrieves relevant nodes from Elasticsearch

2. **Vector Search**
   - Uses a retriever to fetch top-k matching document chunks

3. **Response Generation**
   - Feeds retrieved context into Ollama LLM
   - Streams the answer back to the client

## 🔧 Troubleshooting

### Common Issues

**Issue**: "File already exists" when uploading

- The service prevents duplicate uploads by filename.
- Delete the existing object from MinIO or upload a file with a different name.

**Issue**: Elasticsearch connection failure

- Confirm Elasticsearch is running and reachable.
- Verify `ELASTIC_URL`, `ELASTIC_USERNAME`, and `ELASTIC_PASSWORD` in `.env`.

**Issue**: Ollama model not found

- Ensure the required Ollama models are pulled locally:

```bash
ollama pull all-minilm:latest
ollama pull llama3.2:3b
```

**Issue**: asyncio errors with Python 3.14

- Use Python 3.11 or 3.12 as recommended in `run.md`.

### Debug Mode

Check the `logs/` directory for detailed execution logs from `src/main.py`.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
poetry install --with dev
poetry run pytest
poetry run black .
poetry run isort .
```

## Security Considerations

- Store sensitive credentials in environment variables and keep `.env` out of version control
- Use strong passwords for MinIO and Elasticsearch
- Enable SSL/TLS for production
- Add authentication/authorization to API endpoints for production use
- Keep dependencies updated

## Roadmap

- [ ] Improve query context handling
- [ ] Add authenticated API access
- [ ] Add end-to-end tests for document ingestion

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)

Response:

```json
{ "status": "ok" }
```

#### Upload PDF Document

```bash
curl -X POST http://localhost:3002/upload-pdf \
  -F "file=@/path/to/document.pdf"
```

Response:

```json
{
  "message": "Success, PDF uploaded successfully",
  "filename": "document.pdf"
}
```

#### Query Documents

```bash
curl -X POST http://localhost:3002/prompt \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key points about financial regulations?"}'
```

Response (streaming):

```
data: Based on the documents...

data: The key financial regulations include...

data: [continues streaming]
```

### Python Client Example

```python
import requests

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:3002/upload-pdf",
        files={"file": f}
    )
    print(response.json())

# Query the documents
response = requests.post(
    "http://localhost:3002/prompt",
    json={"query": "Summarize the document"}
)

# Handle streaming response
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
```

## Request Flow

### Upload PDF Flow

1. **File Upload** (`POST /upload-pdf`)
   - Validate file type (PDF only)
   - Read file content
   - Store to MinIO via `DocumentService.store_to_minio()`

2. **Document Parsing** (`DocumentService.parse_document()`)
   - Retrieve PDF from MinIO
   - Convert to markdown using MarkItDown
   - Create LlamaIndex Document with metadata

3. **Embedding Generation**
   - Run ingestion pipeline (Markdown parser → Sentence splitter → Embedding model)
   - Generate embeddings using Ollama
   - Store embeddings in Elasticsearch

### Query Flow

1. **Query Submission** (`POST /prompt`)
   - Receive natural language query
   - Initialize query engine with Elasticsearch index

2. **Vector Search**
   - Generate query embedding
   - Search Elasticsearch for similar document chunks
   - Retrieve top-k results (default: 5)

3. **Response Generation**
   - Pass retrieved context to LLM via Ollama
   - Stream response back to client

## Troubleshooting

### Common Issues

**Issue**: "File already exists" error when uploading

- **Solution**: The system prevents duplicate uploads. Delete the existing file from MinIO or use a different filename.

**Issue**: Elasticsearch connection timeout

- **Solution**: Verify Elasticsearch is running and accessible. Check `ELASTIC_URL` in `.env`.

**Issue**: Ollama model not found

- **Solution**: Ensure required models are pulled: `ollama pull nomic-embed-text` and `ollama pull llama3`

**Issue**: asyncio errors with Python 3.14

- **Solution**: Use Python 3.11 or 3.12 as recommended in `run.md`

### Debug Mode

Enable detailed logging by setting the log level to DEBUG in `src/main.py`. Check the `logs/` directory for detailed execution logs.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Run tests (if available)
poetry run pytest

# Format code
poetry run black .
poetry run isort .
```

## Security Considerations

- Store sensitive credentials (MinIO, Elasticsearch) in environment variables (.env) and add to `.gitignore`
- Use strong passwords for all services
- Enable SSL/TLS for production deployments
- Regularly update dependencies for security patches
- Consider adding authentication/authorization to API endpoints

## Roadmap

- [x] Fix the broken streaming logic
- [x] Integrate a pipecat voice agent module with the RAG pipeline
- [ ] Expose the voice agent via a websocket

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
