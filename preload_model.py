"""
Run this ONCE locally (and it runs automatically on Render during build).
Downloads and caches both sentence transformer models so the app can
run fully offline with HF_HUB_OFFLINE=1.
"""
import os
# Allow downloads during this script
os.environ["HF_HUB_OFFLINE"] = "0"

from sentence_transformers import SentenceTransformer

models = [
    'paraphrase-MiniLM-L3-v2',  # admissions chatbot (app.py)
    'all-MiniLM-L6-v2',          # course chatbot (chatbot_script.py)
]

for model_name in models:
    print(f"Downloading: {model_name} ...")
    SentenceTransformer(model_name)
    print(f"  Cached OK: {model_name}")

print("\nAll models cached. You can now run with HF_HUB_OFFLINE=1.")