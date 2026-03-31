import os
import google.generativeai as genai
from dotenv import load_dotenv

def list_models():
    load_dotenv()
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("GOOGLE_API_KEY not found")
        return
    
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        with open('models_list.txt', 'w') as f:
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    f.write(f"{m.name}\n")
        print("Successfully wrote models to models_list.txt")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
