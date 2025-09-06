#!/bin/bash
# CIRISHome Deployment Script
# Deploys CIRIS engine to Home Assistant Yellow pointing to Jetson Nano

set -e

echo "🏠 CIRISHome Deployment Script"
echo "==============================="

# Check required environment variables
if [ -z "$JETSON_NANO_IP" ]; then
    echo "❌ JETSON_NANO_IP environment variable required"
    echo "   Example: export JETSON_NANO_IP=192.168.1.100"
    exit 1
fi

if [ -z "$HA_TOKEN" ]; then
    echo "❌ HA_TOKEN environment variable required"
    echo "   Get this from Home Assistant -> Profile -> Long-lived access tokens"
    exit 1
fi

if [ -z "$RESPONSIBILITY_ACCEPTED" ]; then
    echo "❌ RESPONSIBILITY_ACCEPTED environment variable required"
    echo "   Set: export RESPONSIBILITY_ACCEPTED=true"
    exit 1
fi

echo "✅ Environment variables validated"
echo "   Jetson Nano IP: $JETSON_NANO_IP"
echo "   HA Token: ${HA_TOKEN:0:10}..."
echo "   Responsibility: $RESPONSIBILITY_ACCEPTED"

# Build the Home Assistant add-on
echo ""
echo "🏗️ Building Home Assistant add-on..."
cd homeassistant-ciris
docker build \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg BUILD_REF=$(git rev-parse HEAD) \
    --tag cirishome-addon:latest \
    .

echo "✅ Docker image built successfully"

# Test the container
echo ""
echo "🧪 Testing container..."
docker run --rm -d \
    -e JETSON_NANO_IP="$JETSON_NANO_IP" \
    -e HA_TOKEN="$HA_TOKEN" \
    -e RESPONSIBILITY_ACCEPTED="$RESPONSIBILITY_ACCEPTED" \
    --name ciris-test \
    cirishome-addon:latest

# Wait for startup
sleep 5

# Basic health check
echo "📋 Container logs:"
docker logs ciris-test

# Cleanup test container
docker stop ciris-test

echo ""
echo "✅ CIRISHome add-on ready for deployment!"
echo ""
echo "📋 Next steps:"
echo "   1. Install add-on in Home Assistant"
echo "   2. Configure Jetson Nano connection"
echo "   3. Set up Voice PE pucks"
echo "   4. Test multi-modal pipeline"
