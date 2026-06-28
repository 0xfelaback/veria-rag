# Veria - RAG Document Query System

A production-ready Retrieval-Augmented Generation (RAG) system that enables intelligent querying of PDF documents using local LLMs and vector embeddings. Veria provides a FastAPI-based REST API for document ingestion and natural language querying.

## 📋 Project Summary

Veria is an intelligent document processing and retrieval system that:

- Ingests PDF documents through a simple upload API
- Parses documents using MarkItDown for accurate text extraction
- Generates embeddings using local models via Ollama
- Stores embeddings in Elasticsearch for efficient similarity search
- Enables natural language querying with context-aware responses

The system is designed for privacy-conscious applications where documents and queries should remain local, leveraging Ollama for local inference and MinIO for scalable document storage.

## ✨ Features

- **PDF Document Ingestion**: Upload and store PDF documents via REST API
- **Intelligent Parsing**: Convert PDFs to markdown using MarkItDown for accurate text extraction
- **Local Embeddings**: Generate vector embeddings using Ollama (no external API dependencies)
- **Vector Search**: Elasticsearch-powered semantic search with configurable similarity thresholds
- **Streaming Responses**: Real-time streaming of LLM responses for better user experience
- **Document Metadata**: Rich metadata tracking including author, category, and parsing timestamps
- **Chunking Strategy**: Configurable document chunking with overlap for better context preservation
- **Comprehensive Logging**: Detailed logging with Loguru for debugging and monitoring
- **Health Checks**: Built-in health check endpoint for monitoring

## 🏗️ Architecture

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
│ - PDF Parsing    │◄────────────────────────────│ - PDF Storage    │
│ - Embedding      │                              │ - Object Mgmt    │
│ - Query Pipeline │                              └──────────────────┘
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
│ - Sentence Split │                              │ - Index Mgmt     │
│ - Embed Model    │                              └──────────────────┘
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

## 📦 Prerequisites

### Required Software

- **Python**: 3.11 or 3.12 (3.14 not recommended due to asyncio issues)
- **Poetry**: For dependency management
- **MinIO**: S3-compatible object storage (local or remote)
- **Elasticsearch**: Vector database for embeddings storage
- **Ollama**: Local LLM and embedding model server

### Ollama Models

Ensure you have the following models pulled in Ollama:

- Embedding model: `nomic-embed-text` or compatible
- LLM model: `llama3` or compatible

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Veria
```

### 2. Install Dependencies

Using Poetry (recommended):

```bash
poetry install
```

Or using pip:

```bash
pip install -e .
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET_NAME=voice-agent-veria-context-documents

# Elasticsearch Configuration
ELASTIC_USERNAME=elastic
ELASTIC_PASSWORD=your_password
ELASTIC_URL=http://localhost:9200
ELASTIC_IGNORE_SSL_ERRORS=true
ES_INDEX_NAME=veria-agent-index
ES_CLUSTER_NAME=veria-agent-cluster
```

### 4. Start Required Services

**MinIO** (if running locally):

```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e MINIO_ROOT_USER=ROOTUSER \
  -e MINIO_ROOT_PASSWORD=MINIOROOTPASSWORD \
  minio/minio server /data --console-address ":9001"
```

**Elasticsearch** (if running locally):

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
# Start the Ollama service
ollama serve
```

## ⚙️ Configuration

### Document Chunking

Configure chunking parameters in `src/Application/services/DocumentService.py`:

```python
sentence_splitter = SentenceSplitter(
    chunk_size=1024,        # Size of each chunk in characters
    chunk_overlap=20,        # Overlap between chunks for context
)
```

### Elasticsearch Index

The system automatically creates the Elasticsearch index on first run. Configure index settings in `src/Infrastructure/elastic_search/index.py`.

### Logging

Logs are stored in the `logs/` directory with timestamp-based filenames. Configure logging in `src/main.py`:

```python
logger.add(
    log_filename,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
)
```

## 🎯 Usage

### Starting the Server

```bash
uvicorn src.main:app --port 3002 --reload --loop asyncio
```

The API will be available at `http://localhost:3002`

### API Endpoints

#### Health Check

```bash
curl http://localhost:3002/health
```

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

## 📊 Request Flow

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

## 🔧 Troubleshooting

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

## 🤝 Contributing

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

## 🔐 Security Considerations

- Store sensitive credentials (MinIO, Elasticsearch) in environment variables (.env) and add to `.gitignore`
- Use strong passwords for all services
- Enable SSL/TLS for production deployments
- Regularly update dependencies for security patches
- Consider adding authentication/authorization to API endpoints

## 🗺️ Roadmap

- [ ] Fix the broken streaming logic
- [ ] Integrate a pipecat voice agent module with the RAG pipeline
- [ ] Expose the voice agent via a websocket

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
