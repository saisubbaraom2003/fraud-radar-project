# app/backend/Dockerfile
# Specifies the environment for the FastAPI backend service.

# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the entire build context (your Git repository) into the container
COPY . .

# Install any needed packages specified in requirements.txt
# Path to requirements.txt is now relative to /usr/src/app
RUN pip install --no-cache-dir --upgrade -r app/backend/requirements.txt

# The command to run when the container launches
# The path to your FastAPI app is now relative to /usr/src/app
CMD uvicorn app.backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
