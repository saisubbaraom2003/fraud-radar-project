# app/backend/Dockerfile
# Specifies the environment for the FastAPI backend service.

# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster
# Using -slim-buster for a smaller image

# Set the working directory in the container
WORKDIR /usr/src/app
# Set a new working directory where the entire repository will be copied

# Copy the entire build context (your Git repository) into the container
# This includes app/backend, models, etc.
COPY . .

# Install any needed packages specified in requirements.txt
# Path to requirements.txt is now relative to /usr/src/app
RUN pip install --no-cache-dir --upgrade -r app/backend/requirements.txt

# The command to run when the container launches
# The path to your FastAPI app is now relative to /usr/src/app
# (i.e., app/backend/app/main.py within the copied repo structure)
CMD uvicorn app.backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
# ${PORT:-8000} means "use PORT env var, or default to 8000 if not set"
# Render automatically sets PORT for web services.
