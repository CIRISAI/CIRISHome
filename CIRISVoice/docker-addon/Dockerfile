ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python and system dependencies
RUN \
    apk add --no-cache \
        python3 \
        py3-pip \
        py3-numpy \
        py3-yaml \
        git \
        ffmpeg \
    && pip3 install --no-cache-dir --upgrade pip

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy and install CIRIS SDK
COPY sdk /app/sdk
RUN cd /app/sdk && pip3 install --no-cache-dir -e .

# Copy the application
COPY src /app/src
COPY config.example.yaml /app/

# Copy run script
COPY run.sh /
RUN chmod a+x /run.sh

# Expose Wyoming port
EXPOSE 10300

# Labels
LABEL \
    io.hass.name="CIRIS Wyoming Bridge" \
    io.hass.description="Connect CIRIS AI to Home Assistant Voice" \
    io.hass.type="addon" \
    io.hass.version="1.0.12" \
    io.hass.arch="aarch64|amd64"

# Run
CMD [ "/run.sh" ]
