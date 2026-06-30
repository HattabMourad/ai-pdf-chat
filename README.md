# AI PDF Chat

Upload any PDF and ask questions about it. Answers are grounded in the document's actual content with source citations.

## How it works

Uses RAG (Retrieval-Augmented Generation):

1. PDF is uploaded and split into chunks
2. Each chunk is embedded using Google's `gemini-embedding-001` model
3. Embeddings are stored in a local ChromaDB vector database
4. On each question, the most relevant chunks are retrieved and sent to `gemini-2.5-flash-lite` to generate a grounded answer

## Tech Stack

- **Backend** — Python, FastAPI
- **AI / Embeddings** — Google Gemini API via LangChain
- **Vector Database** — ChromaDB
- **Frontend** — Plain HTML/CSS/JS

## Getting Started

1. Clone the repo and create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

2. Install dependencies

```bash
pip install fastapi uvicorn python-multipart langchain langchain-google-genai langchain-community langchain-text-splitters chromadb pypdf python-dotenv
```

3. Create a `.env` file

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

5. Open `http://localhost:8000/app` in your browser
