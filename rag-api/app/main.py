from fastapi import FastAPI
from app.api.routes import ingest, query

app = FastAPI(
    title="Enterprise RAG API",
    description="""
## 🚀 Enterprise RAG (Retrieval-Augmented Generation) API

Upload any PDF document and ask natural language questions about it.
The system uses semantic search + LLM generation to answer with source citations.

### How it works:
1. **POST /upload** — Upload a PDF. It gets parsed, chunked, embedded, and stored.
2. **POST /ask** — Ask a question. The system finds relevant chunks and generates a grounded answer.

### Tech Stack:
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB
- **LLM:** LLaMA 3 via Groq API
- **Framework:** FastAPI
    """,
    version="1.0.0",
)

# Register routers
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for monitoring and CI/CD pipelines."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Enterprise RAG API is running!",
        "docs": "/docs",
        "health": "/health",
    }
