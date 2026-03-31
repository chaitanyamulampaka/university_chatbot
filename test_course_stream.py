import asyncio
import os
from dotenv import load_dotenv
from chatbot_script import setup_enhanced_chatbot

async def test_chatbot():
    load_dotenv()
    api_key = os.environ.get('GEMINI_API_KEY')
    print("API KEY loaded:", bool(api_key))
    print("Setting up chatbot CE VR23...")
    try:
        cb = setup_enhanced_chatbot(api_key, 'ce', 'vr23')
        print("Chatbot setup complete. Testing stream_chat...")
        async for c in cb.stream_chat('What are the outcomes?'):
            print(c, end='', flush=True)
        print("\nStream finished successfully.")
    except Exception as e:
        print("\nERROR in stream_chat:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chatbot())
