import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
import ast
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any, AsyncIterator
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
# --- OPTIMIZATION 1: Use Local Embeddings ---
# We replace GoogleGenerativeAIEmbeddings with a local model.
# This avoids a network call for every user query.
from langchain_community.embeddings import SentenceTransformerEmbeddings
# --- END OPTIMIZATION ---

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

# --- Test API Key at Startup ---
def test_api_key():
    """Test the API key by making a simple call to check for quota/permissions."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GOOGLE_API_KEY)
        # Make a minimal call to test
        response = llm.invoke("Test")
        print("API key test successful.")
        return True
    except Exception as e:
        error_str = str(e)
        if "Quota exceeded" in error_str or "limit: 0" in error_str or "Too Many Requests" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("WARNING: API key has quota/permission issues. Admissions chatbot will NOT be disabled entirely.")
            print("Error details:", error_str[:200] + "..." if len(error_str) > 200 else error_str)
            return True # OPTIMIZATION: Return True anyway so we can still use local embeddings
        else:
            print("WARNING: Unexpected API key error. Admissions chatbot will be disabled.")
            print("Error details:", error_str)
            return False

# --- FastAPI App Setup ---
app = FastAPI(title="Admissions Chatbot API")
templates = Jinja2Templates(directory="templates")

# --- Global Variables & Constants ---
KNOWLEDGE_BASE_PATH = 'university_guide.md'
ADMISSIONS_DB_DIR = "admissions_chroma_db" 
vector_store_retriever = None
is_rag_initialized = False

# --- Pydantic Models for Request/Response ---
class AskRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

# We keep the original AskResponse for the /get_suggestions endpoint,
# but the main /stream_ask will return a plain text stream.
class AskResponse(BaseModel):
    answer: str
    suggested_questions: List[str]

# --- Helper Functions (generate_followup_questions is unchanged) ---
def get_default_questions():
    """Returns a list of default questions."""
    return [
        'What courses are offered in Engineering?',
        'What is the fee for an MBA?',
        'What are the eligibility criteria for B.Tech?',
        'Tell me about the scholarship policy',
        'How do I apply for a Ph.D.?'
    ]

def generate_followup_questions(chat_history: List[Dict[str, str]]):
    """Generates context-aware follow-up questions."""
    if not chat_history or not vector_store_retriever:
        return get_default_questions()

    last_user_message = next((item['message'] for item in reversed(chat_history) if item['type'] == 'user'), None)
    if not last_user_message:
        return get_default_questions()

    relevant_docs = vector_store_retriever.invoke(last_user_message)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt_template = f"""
    Based on the provided context from a university admissions guide, suggest 4 short, relevant follow-up questions a prospective student might ask next.
    CRITICAL: The questions MUST be answerable using ONLY the information in the context below. Do not suggest questions if the answer is not in the text.
    Return ONLY a Python-parseable list of strings. For example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]

    Context:
    ---
    {context}
    ---

    Suggested Questions (Python list of strings):
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.6, google_api_key=GOOGLE_API_KEY)
        response = llm.invoke(prompt_template)
        suggested_questions = ast.literal_eval(response.content)
        if isinstance(suggested_questions, list) and all(isinstance(q, str) for q in suggested_questions):
            return suggested_questions
        return get_default_questions()
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing LLM response for follow-up questions: {e}")
        return get_default_questions()
    except Exception as e:
        print(f"Error generating grounded follow-up questions: {e}")
        return get_default_questions()


# --- Core RAG Logic (MODIFIED FOR LOCAL EMBEDDINGS) ---
def initialize_rag_chain():
    """
    Initializes the RAG chain using local, faster embeddings.
    """
    global vector_store_retriever, is_rag_initialized
    try:
        # --- OPTIMIZATION 1: Use Local Embeddings ---
        print("Initializing local sentence transformer embeddings...")
        embeddings = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L3-v2')
        # --- END OPTIMIZATION ---
        
        vector_store = None
        db_file_path = os.path.join(ADMISSIONS_DB_DIR, "chroma.sqlite3")

        if os.path.exists(db_file_path):
            print(f"Loading existing admissions vector store from '{ADMISSIONS_DB_DIR}'...")
            vector_store = Chroma(
                persist_directory=ADMISSIONS_DB_DIR,
                embedding_function=embeddings
            )
            print("Vector store loaded successfully.")
        else:
            print(f"Admissions vector store not found. Creating a new one from '{KNOWLEDGE_BASE_PATH}'...")
            if not os.path.exists(KNOWLEDGE_BASE_PATH):
                print(f"ERROR: Knowledge base file not found at: {KNOWLEDGE_BASE_PATH}")
                is_rag_initialized = False
                return

            loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding='utf-8')
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=ADMISSIONS_DB_DIR
            )
            print(f"New vector store created and saved to '{ADMISSIONS_DB_DIR}'.")

        vector_store_retriever = vector_store.as_retriever()
        is_rag_initialized = True
        print("RAG chain initialized successfully with local embeddings.")

        # Test API key after vector store is ready
        if not test_api_key():
            is_rag_initialized = False
            print("Admissions chatbot disabled due to API key issues.")

    except Exception as e:
        print(f"Error initializing RAG chain: {e}")
        is_rag_initialized = False

# --- OPTIMIZATION 2: Streaming RAG Response ---
async def stream_rag_response(question: str) -> AsyncIterator[str]:
    """
    Generates a streaming response from the RAG chain.
    """
    if not is_rag_initialized or not vector_store_retriever:
        yield "The knowledge base is not yet initialized. Please restart the server."
        return

    template = """
    You are an expert admissions assistant for Siddhartha Academy of Higher Education.
    Your goal is to answer questions accurately based on the provided context.
    If the context doesn't contain the answer, state that you don't have enough information.
    Answer in a clear, friendly, and helpful tone. Format lists or steps clearly if needed.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    
    # Use the standard RAG chain setup
    rag_chain = (
        {"context": vector_store_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        # Use .astream() for an asynchronous stream
        async for chunk in rag_chain.astream(question):
            yield chunk
    except Exception as e:
        # Provide a clearer message for common API quota/permission issues
        error_str = str(e)
        print(f"Error invoking RAG chain stream: {error_str}")
        if "Quota exceeded" in error_str or "limit: 0" in error_str or "Too Many Requests" in error_str:
            yield (
                "Sorry, the AI service is currently unable to process requests due to API quota/permissions. "
                "Please check your Google Cloud billing, quota limits, and that your API key has access to the Gemini API."
            )
        else:
            yield f"Sorry, an error occurred: {error_str}"
# --- END OPTIMIZATION ---


# --- FastAPI Routes (MODIFIED FOR STREAMING) ---

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """Serves the main chat interface."""
    initial_questions = get_default_questions()
    return templates.TemplateResponse("chat.html", {
        "request": request, 
        "suggested_questions": initial_questions,
        "is_rag_initialized": is_rag_initialized
    })

@app.get("/health")
async def health_check():
    """Health check endpoint for Render deployment."""
    return {
        "status": "ok",
        "rag_initialized": is_rag_initialized,
        "service": "university-chatbot"
    }

# --- OPTIMIZATION 2: New /stream_ask endpoint ---
@app.post("/stream_ask")
async def stream_ask(payload: AskRequest):
    """
    Receives a question and returns a streaming response for the answer.
    """
    if not is_rag_initialized:
        return StreamingResponse(
            iter(["Knowledge base not initialized."]), 
            media_type="text/plain"
        )

    return StreamingResponse(
        stream_rag_response(payload.question), 
        media_type="text/plain"
    )
# --- END OPTIMIZATION ---

# --- OPTIMIZATION 2: New /get_suggestions endpoint ---
@app.post("/get_suggestions", response_model=List[str])
async def get_suggestions(payload: AskRequest):
    """
    Receives chat history and returns a list of suggested questions.
    This is called by the frontend *after* the answer stream is complete.
    """
    if not is_rag_initialized:
        return []
    
    return generate_followup_questions(payload.chat_history)
# --- END OPTIMIZATION ---


# --- App startup event ---
@app.on_event("startup")
async def startup_event():
    """
    On startup, find the knowledge base file and initialize the RAG chain.
    """
    print("\n" + "="*50)
    print("🚀 University Chatbot Starting Up")
    print("="*50)
    print(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("Initializing RAG chain...")
    initialize_rag_chain()
    print("✅ Application ready!")
    print("="*50 + "\n")