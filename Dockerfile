# Use a smaller base image
FROM python:3.9.1-slim

# Set work directory
WORKDIR /usr/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libhdf5-dev \
    libhdf5-serial-dev zlib1g-dev libjpeg-dev liblapack-dev libblas-dev libssl-dev libffi-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /usr/app/

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy project files
COPY . /app/
