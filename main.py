from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

# In-memory context storage
context_storage = {"context": None}

# Hugging Face model info
API_KEY = "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"
MODEL = "deepset/roberta-base-squad2"
HF_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Request models
class ContextInput(BaseModel):
    context: str

class QuestionInput(BaseModel):
    question: str

# Endpoint to set context
@app.post("/set-context")
def set_context(data: ContextInput):
    context = data.context.strip()
    word_count = len(context.split())

    if word_count < 50 or word_count > 1000:
        raise HTTPException(
            status_code=400,
            detail="Context must be between 50 and 1000 words."
        )

    context_storage["context"] = context
    return {"message": "Context saved successfully."}

# Optional: Get current context (for debugging)
@app.get("/get-context")
def get_context():
    if context_storage["context"] is None:
        return {"context": "No context set yet."}
    return {"context": context_storage["context"]}

# Endpoint to ask a question
@app.post("/ask")
def ask_question(data: QuestionInput):
    if context_storage["context"] is None:
        raise HTTPException(status_code=400, detail="Context not set.")

    payload = {
        "inputs": {
            "question": data.question,
            "context": context_storage["context"]
        }
    }

    try:
        response = requests.post(HF_URL, headers=headers, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

    if response.status_code != 200:
        try:
            error_detail = response.json().get("error", "Unknown error occurred.")
        except ValueError:
            error_detail = response.text
        raise HTTPException(
            status_code=500,
            detail=f"Hugging Face API error: {error_detail}"
        )

    result = response.json()

    return {"answer": result.get("answer", "No answer found.")}

# Optional: Reset context endpoint
@app.post("/reset-context")
def reset_context():
    context_storage["context"] = None
    return {"message": "Context reset successfully."}
