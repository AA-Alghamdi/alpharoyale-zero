FROM python:3.12-slim

WORKDIR /app

# Install git for weight download
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x download_weights.sh && ./download_weights.sh

EXPOSE 8077
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8077"]
