from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.query_service import answer_question

router = APIRouter()


class QueryRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What is the total revenue mentioned in the document?"}]
        }
    }


@router.post("/ask", summary="Ask a question about uploaded documents")
async def ask_question(request: QueryRequest):
    """
    Ask a natural language question. The system retrieves the most
    relevant chunks from your uploaded documents and uses an LLM
    to generate a grounded, cited answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(question=request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return {"status": "success", "data": result}
