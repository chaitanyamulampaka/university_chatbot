import os
import json
from chatbot_script import EnhancedSyllabusRAGChatbot
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise Exception("❌ GEMINI_API_KEY missing")

# -------- CONFIG --------
DEPARTMENT = "ce"        # change if needed
REGULATION = "vr23"      # change if needed
DATA_ROOT = "data"

# ------------------------

def main():
    print("🚀 Creating vector DB...")

    chatbot = EnhancedSyllabusRAGChatbot(GEMINI_API_KEY)

    # Step 1: Load data (this builds chunks_data)
    data_path = os.path.join(DATA_ROOT, DEPARTMENT, REGULATION)
    syllabus_path = os.path.join(data_path, "syllabus_data.json")
    optimization_path = os.path.join(data_path, "rag_optimization_data.json")

    chatbot.load_data(syllabus_path, optimization_path)

    print(f"✅ Loaded {len(chatbot.chunks_data)} chunks")

    # Step 2: Create DB manually (NO auto-recompute later)
    import chromadb

    DB_PATH = "./chroma_db_by_dept"
    COLLECTION_NAME = f"syllabus_collection_{DEPARTMENT}_{REGULATION}"

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("⚠️ Old collection deleted")
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    # Step 3: Prepare documents (THIS is your answer 🔥)
    documents = [chunk['content'] for chunk in chatbot.chunks_data]

    metadatas = []
    for chunk in chatbot.chunks_data:
        meta = {k: str(v) for k, v in chunk['metadata'].items() if v is not None}
        meta['chunk_type'] = chunk.get('chunk_type', 'unknown')
        meta['source'] = 'syllabus'
        metadatas.append(meta)

    ids = [f"chunk_{i}" for i in range(len(documents))]

    print(f"📦 Total documents: {len(documents)}")

    # Step 4: Generate embeddings (ONLY ONCE HERE)
    embeddings = chatbot.embedding_model.encode(documents).tolist()

    # Step 5: Store in DB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print("✅ Vector DB created successfully!")
    print("📁 Folder: chroma_db_by_dept/")

if __name__ == "__main__":
    main()