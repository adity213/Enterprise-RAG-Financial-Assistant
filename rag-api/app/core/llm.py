from groq import Groq
from app.core.config import settings


class LLMClient:
    """
    Wrapper around the Groq API client.
    
    Why Groq?
    Groq runs LLaMA 3 on custom silicon (LPUs) that are 10x faster than
    standard GPU inference. Their free tier is extremely generous.
    The llama3-8b-8192 model has an 8192 token context window — plenty
    for our retrieved chunks + question.
    
    RAG Prompt Engineering:
    The prompt is carefully structured to:
    1. Give the model a clear ROLE (expert assistant)
    2. Provide the CONTEXT (retrieved chunks) with clear delimiters
    3. State the TASK (answer the question using ONLY the context)
    4. Add a FALLBACK instruction (say "I don't know" if context is insufficient)
    
    This prevents hallucination — the model cannot make up facts
    beyond what is in your document.
    """

    def __init__(self):
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL
        print(f"[LLMClient] Initialized with model: {self._model}")

    def generate_answer(self, question: str, context_chunks: list[dict]) -> str:
        """
        Build a grounded RAG prompt and get an answer from the LLM.
        
        Args:
            question: The user's question
            context_chunks: List of retrieved chunks from the vector store
        
        Returns:
            The LLM's answer string
        """
        # Build context block from retrieved chunks
        context_text = "\n\n---\n\n".join(
            [f"[Source: {c['source']}, Chunk {c['chunk_index']}]\n{c['text']}"
             for c in context_chunks]
        )

        prompt = f"""You are an expert document analyst. Your job is to answer questions 
based STRICTLY on the provided context. Do not use any external knowledge.

CONTEXT:
{context_text}

QUESTION:
{question}

INSTRUCTIONS:
- Answer based ONLY on the context above.
- If the context does not contain enough information to answer, say:
  "I could not find a direct answer in the provided document."
- Be concise and cite which chunk your answer comes from if possible.

ANSWER:"""

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,   # Low temperature = more factual, less creative
            max_tokens=1024,
        )

        return response.choices[0].message.content.strip()


# Singleton instance
llm_client = LLMClient()
