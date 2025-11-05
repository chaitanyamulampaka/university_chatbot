# Step 1: Start from an official Python base image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Step 4: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all your project files into the container
# This includes .py files, .html, .md, and the data/ and static/ directories
COPY . .

# Step 6: Expose the port your app runs on
EXPOSE 8000

# Step 7: Declare volumes for your persistent ChromaDB vector stores
# This prevents your vector data from being lost when the container stops
# and prevents re-building the database on every start.
VOLUME /app/chroma_db_by_dept
VOLUME /app/admissions_chroma_db

# Step 8: Define the command to run your application
# This runs your integrated_main.py using uvicorn, which is
# the same as your `if __name__ == "__main__":` block.
CMD ["uvicorn", "integrated_main:app", "--host", "0.0.0.0", "--port", "8000"]