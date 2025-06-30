# app/backend/Dockerfile
# Specifies the environment for the FastAPI backend service.

# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster
# Using -slim-buster for a smaller image

# Set the working directory in the container
WORKDIR /app # <--- CHANGED: Set working directory to /app for simplicity within container

# Copy the requirements file into the container
COPY app/backend/requirements.txt ./requirements.txt # <--- CHANGED PATH

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the backend application code into the container
COPY app/backend/app ./app # <--- CHANGED PATH

# Copy the models folder from the repository root into the container
# This assumes your models folder is at the root level of your Git repo
COPY models ./models # <--- NEW LINE: To copy models into the container

# The command to run when the container launches
# Uvicorn will bind to 0.0.0.0 and use the PORT environment variable provided by Render.
# Render automatically sets PORT for web services.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
# ${PORT:-8000} means "use PORT env var, or default to 8000 if not set"
# You MUST specify PORT in render.yaml if your app doesn't default to 10000.