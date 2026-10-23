# Use a minimal base image with Python 3.12 installed
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# XGBoost needs the OpenMP runtime, which is not included in the slim image
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install the Python dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the serialized model
COPY . .

# The API listens on port 7860
EXPOSE 7860

# Serve the Flask app (superkart_api inside app.py) with gunicorn
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "app:superkart_api"]
