# Step 1: Start from the Python 3.11 base
FROM python:3.11-slim
WORKDIR /app

# --- THIS IS THE FIX ---

# 1. Install *only* the torch libraries first, forcing pip to
#    use the CPU-only index.
RUN pip --timeout=1000 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Copy the *original* requirements.txt
COPY requirements.txt .

# 3. Now, install all requirements from the default PyPI.
#    This will install 'sentence-transformers' and all other
#    packages. Pip will see torch is already installed and skip it.
RUN pip --timeout=1000 install --no-cache-dir -r requirements.txt
# --- END OF FIX ---

# Step 4: Copy all your project files
COPY . .

# Step 5: Expose the port
EXPOSE 8000

# Step 6: Declare volumes
VOLUME /app/chroma_db_by_dept
VOLUME /app/admissions_chroma_db

# Step 7: Define the command to run your application
CMD ["uvicorn", "integrated_main:app", "--host", "0.0.0.0", "--port", "8000"]