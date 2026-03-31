from sentence_transformers import SentenceTransformer

print("Downloading and caching the model...")
# This forces the model to download and save to the local cache
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
print("Model cached successfully!")