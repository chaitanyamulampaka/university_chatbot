import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# --- OPTIMIZATION: Import StreamingResponse ---
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, AsyncIterator
from dotenv import load_dotenv

# Import from your existing modules
# --- OPTIMIZATION: Import the new stream_chat method ---
from chatbot_script import setup_enhanced_chatbot, EnhancedSyllabusRAGChatbot
import app as admissions_app
# --- END OPTIMIZATION ---

# --- Placements Bot Imports ---
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# --- Lazy startup using lifespan for Render stability ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("🚀 University Chatbot starting (lazy-load mode)")
    print(f"Python: {sys.version.split()[0]}")
    print("All components initialize on first request.")
    print("="*50 + "\n")
    yield

# --- FastAPI App ---
app = FastAPI(
    title="Unified University Chatbot System",
    description="An integrated chatbot for admissions, courses, and placements.",
    version="3.2.0",
    lifespan=lifespan
)

# Serve static files (like logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check (responds immediately, even before init completes) ---
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "admissions_initialized": admissions_app.is_rag_initialized,
        "placements_initialized": placements_agent is not None,
        "course_bots_loaded": len(course_chatbots),
        "service": "university-chatbot"
    }

# --- Request Models ---
class ChatQuery(BaseModel):
    query: str
    department: str
    regulation: Optional[str] = None

class AdmissionsQuery(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

class PlacementsQuery(BaseModel):
    query: str

# --- Global Variables ---
course_chatbots: Dict[str, EnhancedSyllabusRAGChatbot] = {} # Add type hint
placements_agent = None  # Agent for the placements bot
DATA_ROOT_DIRECTORY = "data"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found for course chatbot.")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found for admissions & placements chatbots.")

# --- Placements Bot Logic (Unchanged) ---
# Agent Prefix is copied directly from your script
AGENT_PREFIX = """
You are working with a pandas dataframe in Python. The dataframe is named `df`.
You are a helpful placement assistant designed to answer questions about student placements.
Available columns: academic_year, department, s_no, name, roll_no, branch, company_name, pay_package_lpa
CRITICAL RULES:
1. FOR STRING SEARCHES (like company names):
   - Use case-insensitive containment: df['company_name'].str.contains('VALUE', case=False, na=False)
   - NEVER use exact match (==) for company names
2. PRINT WORKAROUND:
   - ALWAYS cast numbers to string before printing: print(str(value))
   - NEVER print raw numbers or f-strings with numbers
   - For labels, you can print strings directly
3. HUMAN-FRIENDLY OUTPUT:
   - Include student NAME (not just roll number) whenever showing student data
   - Include COMPANY names when relevant
   - Show PACKAGE amounts when discussing placements
   - Use clear separators like "---" between entries
   - Format output with proper spacing and labels
4. FINAL ANSWER FORMAT:
   - Your Final Answer MUST be **identical** to the output from your print statements.
   - Do not add any other text. **Copy the observation to your Final Answer exactly.**
5. EFFICIENT EXECUTION:
   - Execute your code ONCE to get all needed data.
   - After executing your code and seeing the `Observation`, your *only* `Thought` should be:
     `Thought: I have the result. I will now provide this as the Final Answer.`
   - Then, provide the `Final Answer` exactly as it appeared in the `Observation`.
EXAMPLES:
Example 1 - Count query with context:
Question: how many students at tcs ninja
Thought: I need to count students at TCS NINJA using case-insensitive search
Action: python_repl_ast
Action Input: 
count = df[df['company_name'].str.contains('TCS', case=False, na=False) & df['company_name'].str.contains('NINJA', case=False, na=False)].shape[0]
print("🎯 TCS Ninja Placements")
print("=" * 30)
print("Total students placed: " + str(count))
Observation: 🎯 TCS Ninja Placements
==============
Total students placed: 433
Thought: I have the result. I will now provide this as the Final Answer.
Final Answer: 🎯 TCS Ninja Placements
==============
Total students placed: 433
(Other examples omitted for brevity)
Now, begin! Answer questions in a human-friendly way with proper context and formatting.
"""

def initialize_placements_agent():
    """Loads data and initializes the pandas agent for placements."""
    global placements_agent, GOOGLE_API_KEY
    try:
        df = pd.read_csv("placements_data.csv", on_bad_lines='skip')
        
        # Clean up column names
        df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)
        
        # Rename for easier queries
        df = df.rename(columns={
            'companyname': 'company_name',
            'paypackageinlpa': 'pay_package_lpa',
            'sno': 'serial_number'
        })
        
        if 'company_name' in df.columns:
            df['company_name'] = df['company_name'].astype(str)
        if 'pay_package_lpa' in df.columns:
            df['pay_package_lpa'] = pd.to_numeric(df['pay_package_lpa'], errors='coerce')

        print("Placements data loaded and columns cleaned successfully.")

        # Initialize the Language Model
        if not GOOGLE_API_KEY:
            print("ERROR: GOOGLE_API_KEY not set. Placements agent will not be initialized.")
            return

        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)

        # Create the Pandas DataFrame Agent
        placements_agent = create_pandas_dataframe_agent(
            llm,
            df,
            prefix=AGENT_PREFIX,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        print("Placements agent initialized successfully.")

    except FileNotFoundError:
        print("ERROR: 'placements_data.csv' not found. Placements bot will be disabled.")
    except Exception as e:
        print(f"Error initializing placements agent: {e}")

# --- Course Chatbot Endpoints ---
@app.get("/course/departments")
async def get_departments():
    """Returns available departments and regulations for course chatbot."""
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

# --- OPTIMIZATION: Modified for streaming ---
@app.post("/course/chat")
async def handle_course_chat(request: ChatQuery):
    """Handles course/curriculum queries with a streaming response."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server is missing GEMINI API key configuration.")
    
    department = request.department.lower()
    regulation = request.regulation.lower() if request.regulation else None
    user_query = request.query

    chatbot_key = f"{department}_{regulation}" if regulation else department

    if chatbot_key not in course_chatbots:
        # keep only 2 course chatbots in memory to avoid OOM on free tier
        if len(course_chatbots) >= 2:
            oldest_key = next(iter(course_chatbots))
            print(f"Evicting course chatbot '{oldest_key}' to free memory.")
            del course_chatbots[oldest_key]

        try:
            print(f"Loading course chatbot for '{chatbot_key}'...")
            course_chatbots[chatbot_key] = setup_enhanced_chatbot(
                GEMINI_API_KEY, department, regulation, DATA_ROOT_DIRECTORY
            )
            print(f"Course chatbot for '{chatbot_key}' loaded successfully.")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load course chatbot: {e}")

    try:
        chatbot = course_chatbots[chatbot_key]
        # Return a StreamingResponse by calling the new .stream_chat() method
        return StreamingResponse(
            chatbot.stream_chat(user_query),
            media_type="text/plain"
        )
    except Exception as e:
        print(f"Error during course chat streaming: {e}")
        # We can't raise an HTTPException here as the stream has started.
        # The error will be handled inside stream_chat and yielded as text.
        async def error_stream():
            yield f"Sorry, an error occurred: {e}"
        return StreamingResponse(error_stream(), media_type="text/plain")
# --- END OPTIMIZATION ---


# --- Admissions Chatbot Endpoints (MODIFIED FOR STREAMING) ---
@app.post("/admissions/stream_ask")
async def stream_admissions_ask(payload: AdmissionsQuery):
    """
    Receives a question and returns a streaming response for the answer.
    This passes the request directly to the imported admissions_app.
    """
    if not admissions_app.is_rag_initialized:
        print("Lazy-initializing admissions RAG...")
        admissions_app.initialize_rag_chain()

    if not admissions_app.is_rag_initialized:
        return StreamingResponse(
            iter(["Admissions knowledge base failed to initialize. Please try again."]),
            media_type="text/plain"
        )

    return StreamingResponse(
        admissions_app.stream_rag_response(payload.question),
        media_type="text/plain"
    )

@app.post("/admissions/get_suggestions", response_model=List[str])
async def get_admissions_suggestions(payload: AdmissionsQuery):
    """
    Receives chat history and returns a list of suggested questions.
    This passes the request directly to the imported admissions_app.
    """
    if not admissions_app.is_rag_initialized:
        return []
    
    # We don't need to make this async, but we can call it directly
    return admissions_app.generate_followup_questions(payload.chat_history)

@app.get("/admissions/status")
async def get_admissions_status():
    """Returns the initialization status of admissions chatbot."""
    return {"is_initialized": admissions_app.is_rag_initialized}
# --- END OPTIMIZATION ---


# --- Placements Chatbot Endpoint (Unchanged, still blocking) ---
@app.post("/placements/ask")
async def ask_placements_question(request: PlacementsQuery):
    """Handles placements queries using the pandas agent."""
    if not placements_agent:
        print("Lazy-initializing placements agent...")
        initialize_placements_agent()

    if not placements_agent:
        raise HTTPException(
            status_code=503,
            detail="Placements chatbot failed to initialize. Check server logs."
        )

    try:
        response = placements_agent.invoke(request.query)
        answer = response.get('output', 'Sorry, I had trouble processing that request.')
        return {"answer": answer}
    except Exception as e:
        print(f"Error during placements query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --- Main Integration Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serves the integrated chat interface."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "integrated_chat.html")
    if not os.path.exists(html_path):
        html_path = "integrated_chat.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Startup Event removed: Now using lifespan context manager (see above) ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)