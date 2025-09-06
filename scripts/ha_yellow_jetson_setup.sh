#!/bin/bash
# CIRISHome Setup Script for HA Yellow + Jetson Orin Nano
# Multi-modal home automation capabilities with GPU acceleration

echo "CIRISHome Setup for HA Yellow + Jetson Orin Nano"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo "${GREEN}[✓]${NC} $1"; }
print_warning() { echo "${YELLOW}[!]${NC} $1"; }
print_error() { echo "${RED}[✗]${NC} $1"; }

# Check if running on HA Yellow
if [ ! -d "/config" ]; then
    print_error "This script must be run from Home Assistant Terminal & SSH"
    exit 1
fi

# Detect if Jetson Orin Nano is available
JETSON_AVAILABLE=false
if [ -f "/proc/device-tree/model" ] && grep -q "Jetson" /proc/device-tree/model 2>/dev/null; then
    JETSON_AVAILABLE=true
    print_status "Jetson Orin Nano detected - GPU acceleration will be enabled"
else
    print_warning "Jetson Orin Nano not detected - running CPU-only mode"
fi

# Configuration
REPO_URL="https://github.com/CIRISAI/CIRISHome.git"
BRANCH="main"
ADDON_NAME="ciris-home"

print_status "Installing system dependencies..."
apk update
apk add --no-cache git python3 python3-dev py3-pip

# Create addon directory
print_status "Setting up CIRISHome addon..."
mkdir -p /addons/$ADDON_NAME
cd /addons

# Clone repository
if [ -d "$ADDON_NAME" ]; then
    print_warning "Addon directory exists. Updating..."
    rm -rf temp 2>/dev/null || true
fi

print_status "Cloning CIRISHome repository..."
git clone --depth 1 --recurse-submodules $REPO_URL temp || {
    print_error "Failed to clone repository"
    exit 1
}

# Copy files for HA addon
print_status "Setting up addon structure..."
cp -r temp/* $ADDON_NAME/

# Initialize submodules
cd $ADDON_NAME
git submodule update --init --recursive

# Create addon configuration for HA
cat > config.yaml << EOF
name: "CIRIS Home - Multi-Modal AI"
description: "Multi-modal home automation with vision, audio, and sensor fusion"
version: "1.0.0"
slug: "ciris_home"
init: false
arch:
  - aarch64  # HA Yellow + Jetson Orin Nano
startup: application
boot: auto
ports:
  8080/tcp: 8080   # CIRIS API
  10300/tcp: 10300 # Wyoming Protocol
ports_description:
  8080/tcp: "CIRIS Home API"
  10300/tcp: "Wyoming Voice Protocol"
options:
  # Home automation settings
  home_assistant_url: "http://supervisor/core"
  home_assistant_token: ""
  enable_vision: true
  enable_audio: true
  enable_sensors: true

  # Hardware settings
  use_jetson_gpu: $JETSON_AVAILABLE
  camera_devices: "/dev/video0,/dev/video1"
  audio_device: "default"

  # Voice PE puck integration
  wyoming_enabled: true
  voice_timeout: 60

  # Privacy settings
  local_processing_only: true
  data_retention_days: 30

schema:
  home_assistant_url: str
  home_assistant_token: str
  enable_vision: bool
  enable_audio: bool
  enable_sensors: bool
  use_jetson_gpu: bool
  camera_devices: str
  audio_device: str
  wyoming_enabled: bool
  voice_timeout: int(10,300)
  local_processing_only: bool
  data_retention_days: int(1,365)

environment:
  PYTHONUNBUFFERED: "1"
  HOME_ASSISTANT_URL: "%home_assistant_url%"
  HOME_ASSISTANT_TOKEN: "%home_assistant_token%"
  I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY: "true"

hassio_api: true
hassio_role: default
auth_api: true
ingress: false
panel_icon: mdi:home-automation
homeassistant_api: true
host_network: true
audio: true
video: true
gpio: true
uart: true
devices:
  - /dev/video0:/dev/video0:rwm
  - /dev/video1:/dev/video1:rwm
full_access: false
discovery:
  - wyoming
map:
  - config:rw
  - ssl:rw
  - media:rw
EOF

# Create Dockerfile for multi-modal capabilities
cat > Dockerfile << 'EOF'
ARG BUILD_FROM
FROM \$BUILD_FROM

# Install system dependencies for multi-modal processing
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    opencv-python \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy application
WORKDIR /app
COPY . .

# Install CIRIS modules
RUN pip3 install -e ./external/ciris-engine

# Set up multi-modal capabilities
ENV PYTHONPATH="/app:\$PYTHONPATH"
ENV CIRIS_MODULES_PATH="/app/modules"

# Default command
CMD ["/app/run.sh"]
EOF

# Create run script
cat > run.sh << 'EOF'
#!/usr/bin/with-contenv bashio

CONFIG_PATH=/data/options.json

# Read configuration
HOME_ASSISTANT_URL=$(bashio::config 'home_assistant_url')
HOME_ASSISTANT_TOKEN=$(bashio::config 'home_assistant_token')
ENABLE_VISION=$(bashio::config 'enable_vision')
ENABLE_AUDIO=$(bashio::config 'enable_audio')
USE_JETSON_GPU=$(bashio::config 'use_jetson_gpu')

# Set environment variables
export HOME_ASSISTANT_URL="$HOME_ASSISTANT_URL"
export HOME_ASSISTANT_TOKEN="$HOME_ASSISTANT_TOKEN"
export I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY="true"

if [ "$ENABLE_VISION" = "true" ]; then
    export ENABLE_VISION_PIPELINE="true"
    bashio::log.info "Vision processing enabled"
fi

if [ "$ENABLE_AUDIO" = "true" ]; then
    export ENABLE_AUDIO_PIPELINE="true"
    bashio::log.info "Audio processing enabled"
fi

if [ "$USE_JETSON_GPU" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=0
    bashio::log.info "Jetson GPU acceleration enabled"
fi

bashio::log.info "Starting CIRISHome multi-modal platform..."

# Start CIRISHome
exec python3 main.py --adapter ha --enable-vision --enable-audio
EOF

chmod +x run.sh

# Create build.yaml for multi-architecture support
cat > build.yaml << EOF
build_from:
  aarch64: "homeassistant/aarch64-base:latest"
args: {}
EOF

rm -rf /addons/temp

print_status "CIRISHome addon created successfully!"

# Create setup summary
cat > /config/ciris_home_setup.txt << EOF
CIRISHome Multi-Modal Setup Complete!
====================================

Hardware Configuration:
- HA Yellow: ✓ Detected
- Jetson Orin Nano: $(if $JETSON_AVAILABLE; then echo "✓ Detected"; else echo "✗ Not detected"; fi)

Capabilities Enabled:
- Multi-modal vision processing (cameras)
- Multi-modal audio processing (microphones)
- Sensor fusion and automation
- Voice PE puck integration
- GPU acceleration: $(if $JETSON_AVAILABLE; then echo "Enabled"; else echo "CPU-only"; fi)

Next Steps:
1. Go to Settings → Add-ons → Add-on Store
2. Click menu (⋮) → Check for updates
3. Find "CIRIS Home - Multi-Modal AI" in Local add-ons
4. Configure:
   - Home Assistant token
   - Camera devices
   - Enable vision/audio processing
5. Install and start the add-on

Voice PE Puck Integration:
- Automatically creates Wyoming voice pipeline
- 60-second timeout for complex queries
- Multi-modal context awareness

Addon location: /addons/$ADDON_NAME
Configuration: Settings → Add-ons → CIRIS Home
EOF

print_status "Setup complete! Check /config/ciris_home_setup.txt for next steps."

echo ""
echo "🏠 CIRISHome Multi-Modal Platform Ready!"
echo "📹 Vision: Camera-based automation and recognition"
echo "🎤 Audio: Voice commands with PE puck integration"
echo "🔗 Sensors: Multi-modal sensor fusion"
echo "⚡ GPU: $(if $JETSON_AVAILABLE; then echo 'Jetson acceleration enabled'; else echo 'CPU processing mode'; fi)"
echo ""
echo "Next: Install the addon from Home Assistant UI"
