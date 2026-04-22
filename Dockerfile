FROM python:3.11-slim

WORKDIR /app

# System deps for chromadb / builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "agentmesh.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
