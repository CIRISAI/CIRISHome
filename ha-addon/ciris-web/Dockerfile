ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python and dependencies
RUN apk add --no-cache python3 py3-pip && \
    pip3 install --no-cache-dir --break-system-packages aiohttp

# Create app directory
WORKDIR /app

# Copy web server module
COPY rootfs/app/ /app/

# Copy pre-built web assets
COPY www/ /app/web/

# Environment
ENV CIRIS_WEB_ASSETS=/app/web
ENV PYTHONUNBUFFERED=1

# Copy run script
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
