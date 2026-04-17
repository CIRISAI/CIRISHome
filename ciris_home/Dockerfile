# CIRIS Web UI Server
# Serves the KMP web app with HA token injection
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir aiohttp

# Copy web server
COPY webserver.py /app/
COPY installer.py /app/
COPY __init__.py /app/

# Copy web assets (built KMP app)
COPY web/ /app/web/

# Environment
ENV CIRIS_WEB_ASSETS=/app/web
ENV PYTHONUNBUFFERED=1

EXPOSE 8099

CMD ["python", "-m", "webserver", "--host", "0.0.0.0", "--port", "8099"]
