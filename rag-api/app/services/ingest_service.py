import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.embedder import embedder
from app.core.vector_store import vector_store


def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline for a PDF document.
    
    Pipeline Steps:
    1. PARSE:   Extract raw text from every page of the PDF using PyMuPDF
    2. CHUNK:   Split the raw text into overlapping chunks
    3. EMBED:   Convert each chunk into a vector using sentence-transformers
    4. STORE:   Persist vectors + text into ChromaDB
    
    Why chunking with OVERLAP?
    When we split a document into chunks, important information often
    sits at a boundary between two chunks. With overlap (e.g., 50 chars),
    the tail of chunk N is also the head of chunk N+1. This ensures
    no critical sentence is ever split and lost.
    
    Args:
        file_bytes: Raw bytes of the uploaded PDF
        filename: Original filename (used as the collection/document ID)
    
    Returns:
        A dict with ingestion stats
    """
    # --- Step 1: Parse ---
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()
    doc.close()

    if not full_text.strip():
        raise ValueError("Could not extract any text from this PDF. It may be image-based.")

    # --- Step 2: Chunk ---
    # RecursiveCharacterTextSplitter is smarter than a simple splitter:
    # it tries to split on paragraphs first, then sentences, then words.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(full_text)

    # --- Step 3: Embed ---
    print(f"[IngestService] Embedding {len(chunks)} chunks...")
    embeddings = embedder.embed(chunks)

    # --- Step 4: Store ---
    # Use the filename (without extension) as the collection name
    collection_name = "documents"
    doc_id = filename.replace(".pdf", "").replace(" ", "_").lower()

    num_stored = vector_store.add_chunks(
        collection_name=collection_name,
        chunks=chunks,
        embeddings=embeddings,
        doc_id=doc_id,
    )

    return {
        "filename": filename,
        "doc_id": doc_id,
        "total_pages": len(fitz.open(stream=file_bytes, filetype="pdf")),
        "total_chunks": num_stored,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
    }
