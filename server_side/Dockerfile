# People Counting API - Backend Dockerfile
# Optimized for Google Cloud Run

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WEIGHTS_PATH=/app/weights/yolov11n_ncnn_model \
    OUTPUT_DIR=/app/tmp

# Install system dependencies including libGL for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Copy weights
COPY weights/ ./weights/

# Create directories for temporary files
RUN mkdir -p /app/tmp /app/csi_data

# Set working directory to backend
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300
