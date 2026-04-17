ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python, dependencies, and curl for downloading assets
# hadolint ignore=DL3013,DL3018
RUN apk add --no-cache python3 py3-pip curl && \
    pip3 install --no-cache-dir --break-system-packages aiohttp

# Create app directory
WORKDIR /app

# Copy web server module
COPY rootfs/app/ /app/

# Copy pre-built web assets from repo (no fallback)
COPY www/ /app/web/
RUN echo "Web assets:" && ls -la /app/web/ && head -5 /app/web/index.html

# Environment
ENV CIRIS_WEB_ASSETS=/app/web
ENV PYTHONUNBUFFERED=1

# Copy run script
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
