import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"

import ast
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, AsyncIterator
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- Load env ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing")

# --- App ---
app = FastAPI(title="Admissions Chatbot API")

# --- Config ---
ADMISSIONS_DB_DIR = "admissions_chroma_db"
vector_store_retriever = None
is_rag_initialized = False

# --- Models ---
class AskRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

# --- DEFAULT QUESTIONS ---
def get_default_questions():
    return [
        "What courses are offered in Engineering?",
        "What are the eligibility criteria for B.Tech?",
        "Tell me about the scholarship policy",
        "How do I apply for a Ph.D?"
    ]

# --- FOLLOW UPS ---
def generate_followup_questions(chat_history):
    if not chat_history or not vector_store_retriever:
        return get_default_questions()

    last_user = next(
        (item['message'] for item in reversed(chat_history) if item['type'] == 'user'),
        None
    )

    if not last_user:
        return get_default_questions()

    docs = vector_store_retriever.invoke(last_user)
    context = "\n\n".join([d.page_content for d in docs[:3]])

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",
            temperature=0.5,
            google_api_key=GOOGLE_API_KEY
        )

        prompt = f"""
        Suggest 4 short follow-up questions based ONLY on this context:

        {context}

        Return ONLY a Python list.
        """

        res = llm.invoke(prompt)
        return ast.literal_eval(res.content)

    except:
        return get_default_questions()

# --- 🔥 MAIN FIX HERE ---
embedding_model = SentenceTransformerEmbeddings(
    model_name="paraphrase-MiniLM-L3-v2"
)
def initialize_rag_chain():
    global vector_store_retriever, is_rag_initialized

    if is_rag_initialized:
        return

    try:
        print("🚀 Loading precomputed admissions DB...")

        embeddings = embedding_model

        db_file = os.path.join(ADMISSIONS_DB_DIR, "chroma.sqlite3")

        if not os.path.exists(db_file):
            raise Exception("❌ Admissions DB missing.")

        vector_store = Chroma(
            persist_directory=ADMISSIONS_DB_DIR,
            embedding_function=embeddings
        )

        vector_store_retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        is_rag_initialized = True
        print("✅ Admissions DB loaded successfully")

    except Exception as e:
        print(f"❌ RAG init failed: {e}")
        is_rag_initialized = False
# --- STREAM RESPONSE ---
async def stream_rag_response(question: str) -> AsyncIterator[str]:
    if not is_rag_initialized:
        yield "Knowledge base not initialized."
        return

    template = """
    You are a helpful admissions assistant.

    Context:
    {context}

    Question:
    {question}

    Answer clearly:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )

    chain = (
        {"context": vector_store_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        async for chunk in chain.astream(question):
            yield chunk
    except Exception as e:
        yield f"Error: {str(e)}"

# --- ROUTES ---

@app.get("/")
async def home():
    return {"status": "Admissions chatbot running"}

@app.get("/health")
async def health():
    return {"status": "ok", "initialized": is_rag_initialized}

@app.post("/stream_ask")
async def stream_ask(payload: AskRequest):
    global is_rag_initialized

    if not is_rag_initialized:
        initialize_rag_chain()

    if not is_rag_initialized:
        return StreamingResponse(
            iter(["Initialization failed"]),
            media_type="text/plain"
        )

    return StreamingResponse(
        stream_rag_response(payload.question),
        media_type="text/plain"
    )

@app.post("/get_suggestions")
async def suggestions(payload: AskRequest):
    if not is_rag_initialized:
        initialize_rag_chain()

    if not is_rag_initialized:
        return []

    return generate_followup_questions(payload.chat_history)

# --- STARTUP ---
@app.on_event("startup")
async def startup():
    print("\n🚀 Admissions Bot Started (FAST MODE)")
    print("⚡ No embedding recomputation")