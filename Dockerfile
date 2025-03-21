FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenTelemetry for SignOz integration
RUN pip install --no-cache-dir opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp opentelemetry-instrumentation

# Install web dependencies
RUN pip install --no-cache-dir flask flask-wtf

# Copy the rest of the application
COPY . .

# Expose port for web interface
EXPOSE 5000

# Default command - run web server
CMD ["python", "-m", "src.ui.web"]

# CLI is still available using: docker exec -it container_name python -m src.ui.cli interactive
