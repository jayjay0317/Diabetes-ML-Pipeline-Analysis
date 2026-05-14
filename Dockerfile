# Step 1: Python environment
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy all project files
COPY . .

# Step 5: Open Streamlit's default port
EXPOSE 8501

# 6. Run Streamlit application
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]