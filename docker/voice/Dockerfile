FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY setup.py .
RUN pip install -e .

RUN useradd -m -u 1000 wyoming && chown -R wyoming:wyoming /app
USER wyoming

CMD ["python", "-m", "src.bridge"]
