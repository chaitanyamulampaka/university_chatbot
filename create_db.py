import os
import chromadb
from dotenv import load_dotenv
from chatbot_script import EnhancedSyllabusRAGChatbot

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise Exception("❌ GEMINI_API_KEY missing")

# -------- CONFIG --------
DATA_ROOT = "data"
REGULATION = "vr23"

# Automatically detect departments from data folder
DEPARTMENTS = [
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
]

DB_PATH = "./chroma_db_by_dept"

# ------------------------

def create_db_for_department(dept):
    print(f"\n🚀 Processing {dept.upper()}...")

    chatbot = EnhancedSyllabusRAGChatbot(GEMINI_API_KEY)

    data_path = os.path.join(DATA_ROOT, dept, REGULATION)
    syllabus_path = os.path.join(data_path, "syllabus_data.json")
    optimization_path = os.path.join(data_path, "rag_optimization_data.json")

    # Skip if no syllabus file
    if not os.path.exists(syllabus_path):
        print(f"⚠️ Skipping {dept} (no syllabus_data.json)")
        return

    # Load data
    chatbot.load_data(syllabus_path, optimization_path)

    if not chatbot.chunks_data:
        print(f"⚠️ No data for {dept}, skipping...")
        return

    print(f"✅ Loaded {len(chatbot.chunks_data)} chunks")

    # Initialize DB client
    client = chromadb.PersistentClient(path=DB_PATH)

    collection_name = f"syllabus_collection_{dept}_{REGULATION}"

    # Delete old collection if exists
    try:
        client.delete_collection(collection_name)
        print(f"⚠️ Deleted old collection: {collection_name}")
    except:
        pass

    # Create new collection
    collection = client.create_collection(name=collection_name)

    # Prepare documents
    documents = [chunk['content'] for chunk in chatbot.chunks_data]

    metadatas = []
    for chunk in chatbot.chunks_data:
        meta = {k: str(v) for k, v in chunk['metadata'].items() if v is not None}
        meta['chunk_type'] = chunk.get('chunk_type', 'unknown')
        meta['source'] = 'syllabus'
        metadatas.append(meta)

    ids = [f"{dept}_chunk_{i}" for i in range(len(documents))]

    print(f"📦 Total documents: {len(documents)}")

    # Generate embeddings (ONLY ONCE HERE)
    embeddings = chatbot.embedding_model.encode(documents).tolist()

    # Store in DB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ DB created for {dept}!")

# -------- MAIN --------

def main():
    print("\n" + "="*50)
    print("🚀 Creating vector DB for ALL departments")
    print("="*50)

    if not os.path.exists(DATA_ROOT):
        raise Exception("❌ Data folder not found")

    for dept in DEPARTMENTS:
        try:
            create_db_for_department(dept)
        except Exception as e:
            print(f"❌ Error in {dept}: {e}")

    print("\n🎉 ALL DONE!")
    print("📁 DB stored in: chroma_db_by_dept/")

# -------- RUN --------

if __name__ == "__main__":
    main()