from prometheus_client import Counter, Gauge, start_http_server
import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
import psutil
from fastapi import Request
import time

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

def format_image(image):
    'Convert the given image of arbitrary size to 28*28 grayscale image'
    resized_image = image.resize((28, 28)).convert("L")
    image_array = np.array(resized_image)
    flattened_image = image_array.flatten()
    return flattened_image

def predict_digit(data_point):
    'Predict the digit in the given image data point'
    if data_point.size != (28, 28):
        data_point = format_image(data_point).reshape((1, 784))
    else:
        data_point = data_point.convert("L").reshape((1, 784))

    return str(np.random.randint(10))  # Placeholder for predicted digit

def process_memory():
    'Get the memory usage of the current process in kB'
    return psutil.virtual_memory().used/(1024)

@app.post("/predict/")
async def predict_image(request: Request, file: UploadFile = File(...)):
    'Predict the digit in the given image file'

    start_time = time.time()                    # Start time of the API call
    memory_usage_start = process_memory()       # Memory usage before the API call

    contents = await file.read()                # Read the image file contents
    image = Image.open(io.BytesIO(contents))    # Open the image using PIL
    
    # Get client's IP address
    client_ip = request.client.host             # Get the client's IP address
    
    # Update network I/O gauges
    network_io_counters = psutil.net_io_counters()

    predicted_digit = predict_digit(image)      # Predict the digit in the image

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
    start_http_server(5000)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        reload=True,
        workers=1,
        host="0.0.0.0",
        port=8080
    )
