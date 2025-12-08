FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential curl software-properties-common git \
    && rm -rf /var/lib/apt/lists/*

# Copy & install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY agent.py app.py .

# Configure Streamlit
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\nport = 7860\nenableCORS = false\n" > ~/.streamlit/config.toml

# Expose & health check
EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
