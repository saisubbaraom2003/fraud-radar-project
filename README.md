# FraudRadar: Full-Stack Fraud Detection System

This project is an end-to-end full-stack application for detecting fraudulent credit card transactions. It includes a complete ML pipeline for training and evaluation, a backend API to serve the model, and a React frontend for user interaction.

## ✨ Features

- **ML Pipeline Orchestration**: Uses **Prefect** to define and run the entire machine learning workflow.
- **Experiment Tracking**: Leverages **MLflow** to log experiments, compare models, and store artifacts.
- **Cloud Storage**: Integrates with **AWS S3** to store trained models and scalers.
- **Backend API**: A **FastAPI** server that exposes a prediction endpoint.
- **Frontend UI**: A responsive **React** application for real-time predictions.
- **Containerized**: Fully containerized using **Docker** and **Docker Compose** for easy setup and deployment.
- **CI/CD**: Automated testing and deployment pipelines using **GitHub Actions**.

## 📂 Project Structure

<details>
<summary>Click to view the folder structure</summary> 
