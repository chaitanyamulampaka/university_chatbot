import os
import ast
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")


# --- FastAPI App Setup ---
app = FastAPI(title="Admissions Chatbot API")
templates = Jinja2Templates(directory="templates")

# --- Global Variables & Constants ---
KNOWLEDGE_BASE_PATH = 'university_guide.md'
# Define a persistent directory for the admissions DB
ADMISSIONS_DB_DIR = "admissions_chroma_db" 
vector_store_retriever = None
is_rag_initialized = False

# --- Pydantic Models for Request/Response ---
class AskRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

class AskResponse(BaseModel):
    answer: str
    suggested_questions: List[str]

# --- Helper Functions (unchanged) ---
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

    # Use the last user message for context
    last_user_message = next((item['message'] for item in reversed(chat_history) if item['type'] == 'user'), None)
    if not last_user_message:
        return get_default_questions()

    relevant_docs = vector_store_retriever.get_relevant_documents(last_user_message)
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
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.6, google_api_key=GOOGLE_API_KEY)
        response = llm.invoke(prompt_template)
        # Safely evaluate the string to a Python list
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


# --- Core RAG Logic (MODIFIED FOR PERSISTENCE) ---
def initialize_rag_chain():
    """
    Initializes the RAG chain. It loads the vector store from disk if it exists,
    otherwise it creates it from the knowledge base file and saves it to disk.
    """
    global vector_store_retriever, is_rag_initialized
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = None
        
        # This is the specific file ChromaDB creates. We check for this file,
        # not just the directory, to ensure the DB is populated.
        db_file_path = os.path.join(ADMISSIONS_DB_DIR, "chroma.sqlite3")

        if os.path.exists(db_file_path):
            # Load the existing database from disk
            print(f"Loading existing admissions vector store from '{ADMISSIONS_DB_DIR}'...")
            
            # --- ADD THESE 4 LINES ---
            vector_store = Chroma(
                persist_directory=ADMISSIONS_DB_DIR,
                embedding_function=embeddings
            )
            print("Vector store loaded successfully.")
        else:
            # Create and persist the database if it doesn't exist
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
                persist_directory=ADMISSIONS_DB_DIR  # This saves the DB to disk
            )
            print(f"New vector store created and saved to '{ADMISSIONS_DB_DIR}'.")

        vector_store_retriever = vector_store.as_retriever()
        is_rag_initialized = True
        print("RAG chain initialized successfully.")

    except Exception as e:
        print(f"Error initializing RAG chain: {e}")
        is_rag_initialized = False

def get_rag_response(question: str):
    """Generates a response from the RAG chain."""
    if not is_rag_initialized or not vector_store_retriever:
        return "The knowledge base is not yet initialized. Please make sure university_guide.md exists and restart the server."

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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    rag_chain = (
        {"context": vector_store_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        return f"Sorry, an error occurred: {e}"

# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """Serves the main chat interface."""
    initial_questions = get_default_questions()
    return templates.TemplateResponse("chat.html", {
        "request": request, 
        "suggested_questions": initial_questions,
        "is_rag_initialized": is_rag_initialized
    })

@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    """Receives a question, gets an answer from RAG, and generates follow-ups."""
    if not is_rag_initialized:
        return JSONResponse(
            status_code=400,
            content={
                "answer": "Knowledge base not initialized. Please make sure university_guide.md exists and restart the server.",
                "suggested_questions": []
            }
        )

    ai_response = get_rag_response(payload.question)
    
    # Append current exchange to history for generating next suggestions
    current_chat_history = payload.chat_history + [
        {'type': 'user', 'message': payload.question},
        {'type': 'ai', 'message': ai_response}
    ]
    
    suggested_questions = generate_followup_questions(current_chat_history)
    
    return AskResponse(answer=ai_response, suggested_questions=suggested_questions)

# --- App startup event ---
@app.on_event("startup")
async def startup_event():
    """
    On startup, find the knowledge base file and initialize the RAG chain.
    """
    print("Application startup...")
    # The check for the knowledge base file is now inside initialize_rag_chain
    initialize_rag_chain()