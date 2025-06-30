# Use a lightweight Python image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file from your repo
COPY app/backend/requirements.txt ./requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the backend app code
COPY app/backend/app ./app

# Copy the trained model(s)
COPY models ./models

# Command to run your app
CMD ["python", "app/main.py"]
