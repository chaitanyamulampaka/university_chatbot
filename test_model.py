import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

def test_model(model_name):
    print(f"Testing model: {model_name}")
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=os.environ.get('GOOGLE_API_KEY'))
        res = llm.invoke("Say 'Success'")
        print(f"SUCCESS: {model_name} works. Response: {res.content}")
        return True
    except Exception as e:
        print(f"FAILED: {model_name} error: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()
    models_to_try = [
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest"
    ]
    for model in models_to_try:
        if test_model(model):
            break
