import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def process_pdf(file_path: str, session_id: str):
    # Step 1a: Load the PDF and extract raw text
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 1b: Split the text into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # Step 2 + 3: Embed each chunk and store it in Chroma
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"chroma_db/{session_id}",
    )

    return len(chunks)


def ask_question(question: str, session_id: str) -> dict:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    vectorstore = Chroma(
        persist_directory=f"chroma_db/{session_id}",
        embedding_function=embeddings,
    )

    # Step 5: Retrieve the top 4 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Step 6: Ask the LLM to answer using only that context
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    )

    prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [doc.page_content[:200] for doc in retrieved_docs],
    }