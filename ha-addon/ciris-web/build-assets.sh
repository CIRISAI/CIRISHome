#!/bin/bash
# Build or download CIRIS web assets for the HA add-on
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WWW_DIR="$SCRIPT_DIR/www"
MOBILE_WEB_DIR="$SCRIPT_DIR/../../mobile-web"
RELEASE_URL="https://github.com/CIRISAI/CIRISHome/releases/latest/download/ciris-web-assets.tar.gz"

echo "CIRIS Web Assets Builder"
echo "========================"

# Option 1: Build from source
if [ -d "$MOBILE_WEB_DIR/webApp" ]; then
    echo "Found mobile-web directory, building from source..."
    cd "$MOBILE_WEB_DIR"

    if ./gradlew :webApp:wasmJsBrowserDevelopmentExecutableDistribution; then
        echo "Build successful!"
        rm -rf "$WWW_DIR"/*
        cp -r "$MOBILE_WEB_DIR/webApp/build/dist/wasmJs/developmentExecutable/"* "$WWW_DIR/"
        echo "Assets copied to $WWW_DIR"
        exit 0
    else
        echo "Build failed, trying to download release..."
    fi
fi

# Option 2: Download from release
echo "Downloading pre-built assets from release..."
mkdir -p "$WWW_DIR"
cd "$WWW_DIR"

if curl -sL "$RELEASE_URL" | tar xz; then
    echo "Downloaded and extracted assets"
    exit 0
else
    echo "ERROR: Failed to download assets"
    echo ""
    echo "Please either:"
    echo "  1. Build from source: cd mobile-web && ./gradlew :webApp:wasmJsBrowserDevelopmentExecutableDistribution"
    echo "  2. Download manually from: $RELEASE_URL"
    exit 1
fi
