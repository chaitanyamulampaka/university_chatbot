import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Set Your API Key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDg6DDwnsgxJVrZKdm7AWfUtmHyTHWsiE8"

if "YOUR_API_KEY_HERE" in os.environ["GOOGLE_API_KEY"]:
    print("="*50)
    print("ERROR: Please replace 'YOUR_API_KEY_HERE' with your actual Google API key.")
    print("Get one from https://aistudio.google.com/")
    print("="*50)
    exit()

# --- Load and Prepare the Data ---
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
    
    # Ensure company_name is a string
    if 'company_name' in df.columns:
        df['company_name'] = df['company_name'].astype(str)
    
    # Force pay_package_lpa to be numeric
    if 'pay_package_lpa' in df.columns:
        df['pay_package_lpa'] = pd.to_numeric(df['pay_package_lpa'], errors='coerce')

    print("Data loaded and columns cleaned successfully.")
    print("Available columns:", df.columns.tolist())

except FileNotFoundError:
    print("Error: 'placements_data.csv' not found.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Initialize the Language Model ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit()

# --- Create the Pandas DataFrame Agent with IMPROVED PREFIX ---
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
   - Your Final Answer MUST be ONLY the raw output from your print statements
   - Make output conversational and easy to read
   - DO NOT add extra explanatory text in the Final Answer

5. EFFICIENT EXECUTION:
   - Execute your code ONCE to get all needed data
   - After seeing the output ONCE, immediately provide Final Answer
   - DO NOT repeat the same action multiple times

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
==============================
Total students placed: 433
Thought: I have the count. I will provide this as the final answer.
Final Answer: 🎯 TCS Ninja Placements
==============================
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
========================================
Total placement records: 3217
Unique students placed: 1962
Total companies: 343
Average package: 4.89 LPA
Thought: I have all the statistics.
Final Answer: 📊 Placement Overview
========================================
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
========================================
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
Thought: I have all the details.
Final Answer: 🌟 Student with Most Offers
========================================
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
========================================
Company: ACCENTURE
Total placements: 307
Average package: 4.55 LPA
Thought: I have the answer.
Final Answer: 🏢 Top Recruiting Company
========================================
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
========================================
2022-2023 → 1020 placements
2023-2024 → 1097 placements
2024-2025 → 1100 placements
Thought: I have the year-wise breakdown.
Final Answer: 📅 Year-wise Placements
========================================
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
========================================
Student: NIHITHA VEMULAPALLI
Roll No: 208W1A05H5
Company: AMAZON
Package: 52.6 LPA
Department: CSE
Thought: I have complete details.
Final Answer: 💰 Highest Package
========================================
Student: NIHITHA VEMULAPALLI
Roll No: 208W1A05H5
Company: AMAZON
Package: 52.6 LPA
Department: CSE

Now, begin! Answer questions in a human-friendly way with proper context and formatting.
"""

try:
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        prefix=AGENT_PREFIX,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=5,  # Limit iterations to prevent loops
        early_stopping_method="generate"  # Stop when agent thinks it's done
    )
except Exception as e:
    print(f"Error creating agent: {e}")
    exit()

# --- Start the Chat Loop ---
print("\n" + "="*50)
print("Simple Placements Chatbot (Pandas Agent)")
print("Type 'exit' to quit.")
print("="*50)

while True:
    user_question = input("\nAsk your question: ")
    if user_question.lower() == "exit":
        print("Goodbye!")
        break
        
    try:
        response = agent.invoke(user_question)
        answer = response['output']
        
        print("\nFinal Answer:")
        print(answer)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try rephrasing your question.")