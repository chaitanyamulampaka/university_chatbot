import asyncio
import os
from dotenv import load_dotenv
from app import stream_rag_response, initialize_rag_chain

async def test_admissions():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    print("API KEY loaded:", bool(api_key))
    try:
        initialize_rag_chain()
        print("Initialization done. Streaming response...")
        async for chunk in stream_rag_response("What is the fee for B.Tech?"):
            print(chunk, end='', flush=True)
        print("\nStream finished.")
    except Exception as e:
        print("\nERROR in stream_rag_response:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_admissions())
