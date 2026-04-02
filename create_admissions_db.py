from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

KNOWLEDGE_BASE_PATH = "university_guide.md"
DB_DIR = "admissions_chroma_db"

def main():
    print("🚀 Creating admissions vector DB...")

    loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding='utf-8')
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    splits = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(
        model_name='paraphrase-MiniLM-L3-v2'
    )

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("✅ Admissions DB created successfully!")

if __name__ == "__main__":
    main()