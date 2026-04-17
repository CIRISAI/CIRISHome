ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python, dependencies, and curl for downloading assets
RUN apk add --no-cache python3 py3-pip curl && \
    pip3 install --no-cache-dir --break-system-packages aiohttp

# Create app directory
WORKDIR /app

# Copy web server module
COPY rootfs/app/ /app/

# Copy pre-built web assets (or download if empty)
COPY www/ /app/web/
RUN if [ ! -f /app/web/index.html ]; then \
        echo "Downloading web assets from GitHub release..." && \
        curl -sL https://github.com/CIRISAI/CIRISHome/releases/download/v1.0.0-web/ciris-web-assets.tar.gz | tar xz -C /app/web/ && \
        echo "Assets downloaded successfully"; \
    fi

# Environment
ENV CIRIS_WEB_ASSETS=/app/web
ENV PYTHONUNBUFFERED=1

# Copy run script
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
