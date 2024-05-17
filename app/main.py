from prometheus_client import Counter, Gauge, start_http_server
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
import psutil
from fastapi import Request
import time, cv2
from keras.models import Sequential
from keras.models import load_model as keras_model

# Create an instance of the FastAPI class
app = FastAPI()

# Instrument the FastAPI application
Instrumentator().instrument(app).expose(app)

# Prometheus Metrics
REQUEST_COUNTER = Counter('api_requests_total', 'Total number of API requests', ['client_ip'])

RUN_TIME_GAUGE = Gauge('api_run_time_seconds', 'Running time of the API')
TL_TIME_GAUGE = Gauge('api_tl_time_microseconds', 'Effective processing time per character')

MEMORY_USAGE_GAUGE = Gauge('api_memory_usage', 'Memory usage of the API process')
CPU_USAGE_GAUGE = Gauge('api_cpu_usage_percent', 'CPU usage of the API process')

NETWORK_BYTES_SENT_GAUGE = Gauge('api_network_bytes_sent', 'Network bytes sent by the API process')
NETWORK_BYTES_RECV_GAUGE = Gauge('api_network_bytes_received', 'Network bytes received by the API process')

# Load the model
model = keras_model('model/best_model.keras')

def predict_digit(model: Sequential, image: np.ndarray) -> str:
    """
    Predict the digit in the given image using the loaded model.
    
    Args:
        model (Sequential): The loaded Keras Sequential model.
        image (np.ndarray): The input image as a NumPy array.
        
    Returns:
        str: The predicted digit as a string.
    """
    # Preprocess the image
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = image.reshape(1, 784) / 255.0  # Reshape and normalize pixel values

    # Make prediction
    prediction = model.predict(image)
    digit = str(np.argmax(prediction))
    return digit

def process_memory():
    'Get the memory usage of the current process in kB'
    return psutil.virtual_memory().used/(1024)

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    'Predict the digit in the given image file'

    start_time = time.time()                    # Start time of the API call
    memory_usage_start = process_memory()       # Memory usage before the API call

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get client's IP address
    client_ip = request.client.host             # Get the client's IP address
    
    # Update network I/O gauges
    network_io_counters = psutil.net_io_counters()
    
    predicted_digit = predict_digit(model, image)      # Predict the digit in the image

    cpu_percent = psutil.cpu_percent(interval=1)    # Get the CPU usage percentage
    memory_usage_end = process_memory()             # Get the memory usage after the API call

    CPU_USAGE_GAUGE.set(cpu_percent)                                            # Set the CPU usage gauge
    MEMORY_USAGE_GAUGE.set((np.abs(memory_usage_end-memory_usage_start)))       # Set the memory usage gauge
    NETWORK_BYTES_SENT_GAUGE.set(network_io_counters.bytes_sent)                # Set the network bytes sent gauge
    NETWORK_BYTES_RECV_GAUGE.set(network_io_counters.bytes_recv)                # Set the network bytes received gauge
    
    # Calculate API running time
    end_time = time.time()
    run_time = end_time - start_time
    
    # Record API usage metrics
    REQUEST_COUNTER.labels(client_ip).inc()         # Increment the request counter             
    RUN_TIME_GAUGE.set(run_time)                    # Set the running time gauge
    
    # Calculate T/L time
    input_length = len(contents)
    tl_time = (run_time / input_length) * 1e6   # microseconds per pixel
    TL_TIME_GAUGE.set(tl_time)                  # Set the T/L time gauge
    
    return {"digit": predicted_digit}

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        reload=True,
        workers=1,
        host="127.0.0.1",
        port=8001
    )
