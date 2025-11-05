import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Import from your existing modules
from chatbot_script import setup_enhanced_chatbot
import app as admissions_app

# --- Placements Bot Imports ---
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="Unified University Chatbot System",
    description="An integrated chatbot for admissions, courses, and placements.",
    version="3.1.0"
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
course_chatbots = {}
placements_agent = None  # Agent for the placements bot
DATA_ROOT_DIRECTORY = "data"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found for course chatbot.")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found for admissions & placements chatbots.")

# --- Placements Bot Logic (from placement_bot.py) ---

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

Example 2 - Overall placement summary:
Question: how are placements
Thought: I need total placements, unique students, unique companies, and average package
Action: python_repl_ast
Action Input:
print("📊 Placement Overview")
print("=" * 40)
print("Total placement records: " + str(df.shape[0]))
print("Unique students placed: " + str(df['roll_no'].nunique()))
print("Total companies: " + str(df['company_name'].nunique()))
print("Average package: " + str(round(df['pay_package_lpa'].mean(), 2)) + " LPA")
Observation: 📊 Placement Overview
==============
Total placement records: 3217
Unique students placed: 1962
Total companies: 343
Average package: 4.89 LPA
Thought: I have the result. I will now provide this as the Final Answer.
Final Answer: 📊 Placement Overview
==============
Total placement records: 3217
Unique students placed: 1962
Total companies: 343
Average package: 4.89 LPA

Example 3 - Student with most offers (SHOW NAME + COMPANIES):
Question: which student got more offers
Thought: I need to find student with most offers and show their details
Action: python_repl_ast
Action Input:
student_counts = df['roll_no'].value_counts()
top_roll = student_counts.index[0]
top_count = student_counts.iloc[0]
student_data = df[df['roll_no'] == top_roll]
student_name = student_data['name'].iloc[0]
companies = student_data['company_name'].tolist()

print("🌟 Student with Most Offers")
print("=" * 40)
print("Name: " + student_name)
print("Roll No: " + top_roll)
print("Total offers: " + str(top_count))
print("\nCompanies:")
for i, company in enumerate(companies, 1):
    print(str(i) + ". " + company)
Observation: 🌟 Student with Most Offers
==============
Name: NALLABOTHULA UPENDRA
Roll No: 228W5A0237
Total offers: 10

Companies:
1. KIA INDIA PVT LTD
2. EDIGLOBE
3. INDWELL
4. VOLTECH
5. SUNTEK
...
Thought: I have the result. I will now provide this as the Final Answer.
Final Answer: 🌟 Student with Most Offers
==============
Name: NALLABOTHULA UPENDRA
Roll No: 228W5A0237
Total offers: 10

Companies:
1. KIA INDIA PVT LTD
2. EDIGLOBE
3. INDWELL
4. VOLTECH
5. SUNTEK
...

Example 4 - Top company with more details:
Question: which company is hiring the most
Thought: I need top company with placement details
Action: python_repl_ast
Action Input:
company_counts = df['company_name'].value_counts()
top_company = company_counts.index[0]
top_count = company_counts.iloc[0]
company_data = df[df['company_name'].str.contains(top_company, case=False, na=False)]
avg_package = company_data['pay_package_lpa'].mean()

print("🏢 Top Recruiting Company")
print("=" * 40)
print("Company: " + top_company)
print("Total placements: " + str(top_count))
print("Average package: " + str(round(avg_package, 2)) + " LPA")
Observation: 🏢 Top Recruiting Company
==============
Company: ACCENTURE
Total placements: 307
Average package: 4.55 LPA
Thought: I have the result. I will now provide this as the Final Answer.
Final Answer: 🏢 Top Recruiting Company
==============
Company: ACCENTURE
Total placements: 307
Average package: 4.55 LPA

Example 5 - Year-wise with better formatting:
Question: give year wise placements
Thought: I need to show placements by year with good formatting
Action: python_repl_ast
Action Input:
print("📅 Year-wise Placements")
print("=" * 40)
for year, count in df.groupby('academic_year').size().items():
    print(year + " → " + str(count) + " placements")
Observation: 📅 Year-wise Placements
==============
2022-2023 → 1020 placements
2023-2024 → 1097 placements
2024-2025 → 1100 placements
Thought: I have the result. I will now provide this as the Final Answer.
Final Answer: 📅 Year-wise Placements
==============
2022-2023 → 1020 placements
2023-2024 → 1097 placements
2024-2025 → 1100 placements

Example 6 - Highest package with student details:
Question: who got highest package
Thought: I need to find student with highest package and show details
Action: python_repl_ast
Action Input:
max_package = df['pay_package_lpa'].max()
top_student = df[df['pay_package_lpa'] == max_package].iloc[0]

print("💰 Highest Package")
print("=" * 40)
print("Student: " + top_student['name'])
print("Roll No: " + top_student['roll_no'])
print("Company: " + top_student['company_name'])
print("Package: " + str(max_package) + " LPA")
print("Department: " + top_student['branch'])
Observation: 💰 Highest Package
==============
Student: NIHITHA VEMULAPALLI
Roll No: 208W1A05H5
Company: AMAZON
Package: 52.6 LPA
Department: CSE
Thought: I have the result. I will now provide this as the Final Answer.
Final Answer: 💰 Highest Package
==================
Student: NIHITHA VEMULAPALLI
Roll No: 208W1A05H5
Company: AMAZON
Package: 52.6 LPA
Department: CSE

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

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, google_api_key=GOOGLE_API_KEY)

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

@app.post("/course/chat")
async def handle_course_chat(request: ChatQuery):
    """Handles course/curriculum queries."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server is missing GEMINI API key configuration.")
    
    department = request.department.lower()
    regulation = request.regulation.lower() if request.regulation else None
    user_query = request.query

    chatbot_key = f"{department}_{regulation}" if regulation else department

    if chatbot_key not in course_chatbots:
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
        response = chatbot.chat(user_query)
        return response
    except Exception as e:
        print(f"Error during course chat processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

# --- Admissions Chatbot Endpoints ---
@app.post("/admissions/ask")
async def ask_admissions_question(payload: AdmissionsQuery):
    """Handles admissions queries."""
    if not admissions_app.is_rag_initialized:
        raise HTTPException(
            status_code=400,
            detail="Admissions knowledge base not initialized."
        )

    ai_response = admissions_app.get_rag_response(payload.question)
    
    current_chat_history = payload.chat_history + [
        {'type': 'user', 'message': payload.question},
        {'type': 'ai', 'message': ai_response}
    ]
    
    suggested_questions = admissions_app.generate_followup_questions(current_chat_history)
    
    return {
        "answer": ai_response,
        "suggested_questions": suggested_questions
    }

@app.get("/admissions/status")
async def get_admissions_status():
    """Returns the initialization status of admissions chatbot."""
    return {"is_initialized": admissions_app.is_rag_initialized}

# --- Placements Chatbot Endpoint (NEW) ---
@app.post("/placements/ask")
async def ask_placements_question(request: PlacementsQuery):
    """Handles placements queries using the pandas agent."""
    if not placements_agent:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Placements chatbot is not initialized. Check server logs."
        )
    
    try:
        # The agent's 'invoke' method runs the query
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
    with open("integrated_chat.html", "r", encoding="utf-8") as f:
        return f.read()

# --- Startup Event (Updated) ---
@app.on_event("startup")
async def startup_event():
    """Initialize admissions and placements chatbots on startup."""
    print("Starting Unified Chatbot System...")
    
    # Initialize admissions chatbot
    if os.path.exists(admissions_app.KNOWLEDGE_BASE_PATH):
        print(f"Initializing admissions chatbot from '{admissions_app.KNOWLEDGE_BASE_PATH}'...")
        admissions_app.initialize_rag_chain()
    else:
        print(f"Warning: Admissions knowledge base not found at '{admissions_app.KNOWLEDGE_BASE_PATH}'")
    
    # Initialize placements chatbot
    print("Initializing placements chatbot...")
    initialize_placements_agent()
    
    print("Unified Chatbot System ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)