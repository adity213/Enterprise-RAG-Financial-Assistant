from app.core.config import settings
from app.core.embedder import embedder
from app.core.vector_store import vector_store
from app.core.llm import llm_client


def answer_question(question: str) -> dict:
    """
    Full RAG query pipeline.
    
    Pipeline Steps:
    1. EMBED QUERY:  Convert the user's question into a vector
    2. RETRIEVE:     Find the top-K most similar chunks from ChromaDB
    3. AUGMENT:      Inject retrieved chunks into the LLM prompt as context
    4. GENERATE:     Get the LLM's grounded answer
    
    This is the core of RAG (Retrieval-Augmented Generation):
    - "Retrieval" = Step 1 + 2
    - "Augmented" = Step 3 (we augment the LLM's knowledge with our docs)
    - "Generation" = Step 4
    
    Args:
        question: The user's natural language question
    
    Returns:
        A dict with the answer and the source chunks used
    """
    # --- Step 1: Embed the query ---
    # We use the SAME embedding model as during ingestion.
    # This is critical: if you embed documents with model A and queries with
    # model B, the vector spaces won't align and retrieval will be garbage.
    query_embedding = embedder.embed([question])[0]

    # --- Step 2: Retrieve top-K similar chunks ---
    context_chunks = vector_store.query(
        collection_name="documents",
        query_embedding=query_embedding,
        top_k=settings.TOP_K_RESULTS,
    )

    if not context_chunks:
        return {
            "question": question,
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    # --- Step 3 + 4: Augment and Generate ---
    answer = llm_client.generate_answer(question=question, context_chunks=context_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "similarity_score": c["similarity_score"],
                "text_preview": c["text"][:200] + "...",  # First 200 chars for preview
            }
            for c in context_chunks
        ],
    }
