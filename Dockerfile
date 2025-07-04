# Use Python 3.8 base image
FROM python:3.8-slim

# Install system packages for OpenCV and mediapipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port your app runs on
EXPOSE 5000

# Start the Flask app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
