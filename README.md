# Image Digit Prediction with Prometheus Monitoring
This project implements a FastAPI application that takes an uploaded image file, predicts the digit present in the image using a pre-trained model, and returns the predicted digit as a response. Additionally, the application utilizes Prometheus for detailed monitoring of various performance metrics.

# Project Structure
The project consists of the following files:

app/main.py: The main FastAPI application script.
model/best_model.keras: The pre-trained Keras model for digit prediction.

# Setting Up the Project
## Install Dependencies:
Use `pip install -r requirements.txt`

## Replace the Pre-trained Model:
Place your pre-trained Keras model for digit prediction in the model directory as best_model.keras. Ensure the model is compatible with Keras and expects a pre-processed image as input.

# Running the Application

## Start Prometheus Server:
- Open a separate terminal window and run the following command to start the Prometheus metrics server: `python main.py`
- This will start the Prometheus server on port **8000**.

## Run the FastAPI Application:
- In another terminal window, navigate to the project directory and run: `uvicorn app.main:app --reload`
- This will start the FastAPI application on port 8001.

## Accessing the API

Body: Upload an image file containing a single digit.
The API will respond with a JSON object containing the predicted digit:

JSON
{
  "digit": "7"
}

# Monitoring with Prometheus
Prometheus exposes various metrics related to the API's performance. You'll need a separate Prometheus instance and a suitable visualization tool (e.g., Grafana) to view these metrics. Configure Prometheus to scrape targets from port `8000` (where the Prometheus server runs in this project).

Here are some of the exposed metrics:

- api_requests_total: Total number of API requests received.
- api_run_time_seconds: Running time of the API per request.
- api_tl_time_microseconds: Effective processing time per character in the input image.
- api_memory_usage: Memory usage of the API process.
- api_cpu_usage_percent: CPU usage of the API process.
