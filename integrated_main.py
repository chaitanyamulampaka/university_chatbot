import os
import sys

# HF_HUB_OFFLINE is set by Render's environment variables (render.yaml).
# We do NOT scan the filesystem here — that caused slow startup and port-bind
# timeouts. If running locally without the env var, downloads are allowed.
_hf_offline = os.environ.get("HF_HUB_OFFLINE", "0")
print(f"HF_HUB_OFFLINE = {_hf_offline}")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, AsyncIterator
from dotenv import load_dotenv

load_dotenv()

# ── Safe imports with explicit error messages ─────────────────────────────────
try:
    from chatbot_script import setup_enhanced_chatbot, EnhancedSyllabusRAGChatbot
    print("chatbot_script imported OK")
except Exception as _e:
    print(f"FATAL: Failed to import chatbot_script: {_e}")
    raise

try:
    import app as admissions_app
    print("admissions app imported OK")
except Exception as _e:
    print(f"FATAL: Failed to import admissions app: {_e}")
    raise

try:
    import pandas as pd
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("pandas / langchain imports OK")
except Exception as _e:
    print(f"FATAL: Failed to import pandas/langchain: {_e}")
    raise

# ── Environment keys ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set")

# ── safe_stream helper ────────────────────────────────────────────────────────
async def safe_stream(generator):
    async for chunk in generator:
        if isinstance(chunk, dict):
            chunk = chunk.get("text", "")
        elif isinstance(chunk, list):
            chunk = " ".join(
                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                for c in chunk
            )
        elif hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, dict):
                chunk = content.get("text", "")
            elif isinstance(content, list):
                chunk = " ".join(
                    c.get("text", str(c)) if isinstance(c, dict) else str(c)
                    for c in content
                )
            else:
                chunk = str(content)
        if chunk is None:
            continue
        yield str(chunk)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    print("="*50)
    print("University Chatbot starting (lazy-load mode)")
    print(f"Python: {sys.version.split()[0]}")
    print("="*50)
    yield

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Unified University Chatbot System",
    version="3.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Models ────────────────────────────────────────────────────────────
class ChatQuery(BaseModel):
    query: str
    department: str
    regulation: Optional[str] = None

class AdmissionsQuery(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

class PlacementsQuery(BaseModel):
    query: str

# ── Global state ──────────────────────────────────────────────────────────────
course_chatbots: Dict[str, EnhancedSyllabusRAGChatbot] = {}
placements_agent = None
DATA_ROOT_DIRECTORY = "data"

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "admissions_initialized": admissions_app.is_rag_initialized,
        "placements_initialized": placements_agent is not None,
        "course_bots_loaded": len(course_chatbots),
        "service": "university-chatbot"
    }

# ── Placements agent ──────────────────────────────────────────────────────────
AGENT_PREFIX = """
You are working with a pandas dataframe in Python. The dataframe is named `df`.
You are a helpful placement assistant designed to answer questions about student placements.
Available columns: academic_year, department, s_no, name, roll_no, branch, company_name, pay_package_lpa
CRITICAL RULES:
1. Use case-insensitive containment for string searches.
2. ALWAYS cast numbers to string before printing.
3. Include student NAME, COMPANY, and PACKAGE in output.
4. Final Answer MUST match print output exactly.
5. Execute code ONCE then give Final Answer.
Now begin!
"""

def initialize_placements_agent():
    global placements_agent
    try:
        df = pd.read_csv("placements_data.csv", on_bad_lines='skip')
        df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)
        df = df.rename(columns={
            'companyname': 'company_name',
            'paypackageinlpa': 'pay_package_lpa',
            'sno': 'serial_number'
        })
        if 'company_name' in df.columns:
            df['company_name'] = df['company_name'].astype(str)
        if 'pay_package_lpa' in df.columns:
            df['pay_package_lpa'] = pd.to_numeric(df['pay_package_lpa'], errors='coerce')
        print("Placements data loaded")
        if not GOOGLE_API_KEY:
            print("GOOGLE_API_KEY missing - placements disabled")
            return
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )
        placements_agent = create_pandas_dataframe_agent(
            llm, df,
            prefix=AGENT_PREFIX,
            verbose=False,
            allow_dangerous_code=True,
            max_iterations=5,
        )
        print("Placements agent initialized")
    except FileNotFoundError:
        print("placements_data.csv not found - placements disabled")
    except Exception as e:
        print(f"Placements agent init error: {e}")

# ── Course chatbot endpoints ──────────────────────────────────────────────────
@app.get("/course/departments")
async def get_departments():
    departments_with_regulations = {}
    if not os.path.exists(DATA_ROOT_DIRECTORY):
        raise HTTPException(status_code=404, detail="Data directory not found.")
    try:
        for dept in os.listdir(DATA_ROOT_DIRECTORY):
            dept_path = os.path.join(DATA_ROOT_DIRECTORY, dept)
            if os.path.isdir(dept_path):
                subdirs = [d for d in os.listdir(dept_path) if os.path.isdir(os.path.join(dept_path, d))]
                if subdirs and any("syllabus_data.json" in os.listdir(os.path.join(dept_path, sd)) for sd in subdirs):
                    departments_with_regulations[dept] = subdirs
                elif "syllabus_data.json" in os.listdir(dept_path):
                    departments_with_regulations[dept] = []
        return {"departments": departments_with_regulations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning directories: {e}")

@app.post("/course/chat")
async def handle_course_chat(request: ChatQuery):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    department = request.department.lower()
    regulation = request.regulation.lower() if request.regulation else None
    chatbot_key = f"{department}_{regulation}" if regulation else department
    if chatbot_key not in course_chatbots:
        if len(course_chatbots) >= 2:
            oldest_key = next(iter(course_chatbots))
            del course_chatbots[oldest_key]
        try:
            course_chatbots[chatbot_key] = setup_enhanced_chatbot(
                GEMINI_API_KEY, department, regulation, DATA_ROOT_DIRECTORY
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load course chatbot: {e}")
    try:
        chatbot = course_chatbots[chatbot_key]
        return StreamingResponse(safe_stream(chatbot.stream_chat(request.query)), media_type="text/plain")
    except Exception as e:
        async def error_stream(msg):
            yield f"Sorry, an error occurred: {msg}"
        return StreamingResponse(error_stream(str(e)), media_type="text/plain")

# ── Admissions endpoints ──────────────────────────────────────────────────────
@app.post("/admissions/stream_ask")
async def stream_admissions_ask(payload: AdmissionsQuery):
    if not admissions_app.is_rag_initialized:
        admissions_app.initialize_rag_chain()
    if not admissions_app.is_rag_initialized:
        return StreamingResponse(iter(["Admissions knowledge base failed to initialize."]), media_type="text/plain")
    return StreamingResponse(admissions_app.stream_rag_response(payload.question), media_type="text/plain")

@app.post("/admissions/get_suggestions", response_model=List[str])
async def get_admissions_suggestions(payload: AdmissionsQuery):
    if not admissions_app.is_rag_initialized:
        return []
    return admissions_app.generate_followup_questions(payload.chat_history)

@app.get("/admissions/status")
async def get_admissions_status():
    return {"is_initialized": admissions_app.is_rag_initialized}

# ── Placements endpoint ───────────────────────────────────────────────────────
@app.post("/placements/ask")
async def ask_placements_question(request: PlacementsQuery):
    if not placements_agent:
        initialize_placements_agent()
    if not placements_agent:
        raise HTTPException(status_code=503, detail="Placements chatbot failed to initialize.")
    try:
        response = placements_agent.invoke(request.query)
        return {"answer": response.get('output', 'Sorry, I had trouble processing that.')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# ── Main page ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    html_path = os.path.join(os.path.dirname(__file__), "integrated_chat.html")
    if not os.path.exists(html_path):
        html_path = "integrated_chat.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)