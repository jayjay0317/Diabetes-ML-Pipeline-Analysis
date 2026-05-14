# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# 6. Make port 5000 available to the world outside this container
EXPOSE 5000

# 7. Run the application
CMD ["Python", "src/app.py"]