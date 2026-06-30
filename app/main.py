import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.rag import process_pdf, ask_question

app = FastAPI(title="AI PDF Chat")

# Allow the frontend to call this API from a different port/domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # we'll restrict this later for production
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks: int

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate a unique session ID for this PDF
    session_id = str(uuid.uuid4())

    # Save the uploaded file temporarily
    upload_path = f"app/uploads/{session_id}.pdf"
    os.makedirs("app/uploads", exist_ok=True)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunk_count = process_pdf(upload_path, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up the uploaded file — we only need the embeddings now
        os.remove(upload_path)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        chunks=chunk_count,
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    try:
        result = ask_question(request.question, request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
    )

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI PDF Chat API is running"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
async def serve_frontend():
    return FileResponse("static/index.html")