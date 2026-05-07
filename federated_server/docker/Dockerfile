# Flower Federated Learning Server
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (full stack for central server)
COPY requirements-flower.txt /app/
RUN pip install --upgrade pip && pip install -r requirements-flower.txt

# Copy project
COPY . /app/

# Expose Flower server port
EXPOSE 8080

# Set default environment variables for Flower
ENV FLOWER_HOST=0.0.0.0
ENV FLOWER_PORT=8080
ENV FLOWER_ROUNDS=5

# Run Flower server
CMD ["python", "-u", "flower_server/server.py"]
