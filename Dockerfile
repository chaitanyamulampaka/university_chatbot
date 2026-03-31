# Use Python 3.10 slim as the base image (includes SQLite >= 3.35 for ChromaDB)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages
# build-essential: for compiling some python extensions
# curl: useful for healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
COPY preload_model.py .
# Run it immediately to cache the model inside the image
RUN python preload_model.py
# Copy the rest of the application code
COPY . .

# Create directories for data persistence if they don't exist
RUN mkdir -p admissions_chroma_db chroma_db_by_dept data

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application using Uvicorn
# Cloud Run sets the PORT environment variable, so we must use it.
CMD ["sh", "-c", "uvicorn integrated_main:app --host 0.0.0.0 --port ${PORT:-8000}"]