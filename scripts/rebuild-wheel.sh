#!/bin/bash
# rebuild-wheel.sh
# Completely rebuilds the CIRIS Agent wheel from scratch
#
# Usage: ./scripts/rebuild-wheel.sh [--deploy]
#   --deploy    Also deploy the wheel to Home Assistant after building

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CIRIS_AGENT_DIR="${CIRIS_AGENT_DIR:-$(cd "$REPO_ROOT/.." && pwd)/CIRISAgent}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Parse arguments
DEPLOY=false
for arg in "$@"; do
    case $arg in
        --deploy)
            DEPLOY=true
            ;;
    esac
done

# Check CIRISAgent directory exists
if [ ! -d "$CIRIS_AGENT_DIR" ]; then
    log_error "CIRISAgent directory not found: $CIRIS_AGENT_DIR"
    log_info "Set CIRIS_AGENT_DIR environment variable to override"
    exit 1
fi

log_info "Building wheel from: $CIRIS_AGENT_DIR"

# Step 1: Clean previous builds
log_step "Cleaning previous builds..."
rm -rf "$CIRIS_AGENT_DIR/dist" "$CIRIS_AGENT_DIR/build" "$CIRIS_AGENT_DIR/*.egg-info" 2>/dev/null || true
rm -rf "$CIRIS_AGENT_DIR/src/ciris_engine/*.egg-info" 2>/dev/null || true

# Step 2: Build the wheel
log_step "Building wheel..."
cd "$CIRIS_AGENT_DIR"

# Use uv if available, otherwise pip
if command -v uv &> /dev/null; then
    log_info "Using uv for faster builds"
    uv build --wheel
else
    log_info "Using pip (install uv for faster builds: pip install uv)"
    python -m pip wheel . --no-deps -w dist
fi

# Step 3: Verify the wheel was created
WHEEL=$(ls "$CIRIS_AGENT_DIR/dist"/ciris_agent-*.whl 2>/dev/null | sort -V | tail -1)
if [ -z "$WHEEL" ]; then
    log_error "Wheel build failed - no wheel file found in dist/"
    exit 1
fi

WHEEL_NAME=$(basename "$WHEEL")
WHEEL_VERSION=$(echo "$WHEEL_NAME" | sed 's/ciris_agent-\([^-]*\)-.*/\1/')

log_info "Wheel built successfully: $WHEEL_NAME"
log_info "Version: $WHEEL_VERSION"

# Step 4: Copy wheel to local addon directory
log_step "Copying wheel to local addon directory..."
cp "$WHEEL" "$REPO_ROOT/ciris-agent/"
log_info "Copied to: $REPO_ROOT/ciris-agent/$WHEEL_NAME"

# Step 5: Deploy if requested
if [ "$DEPLOY" = true ]; then
    log_step "Deploying wheel to Home Assistant..."
    "$SCRIPT_DIR/deploy-agent-wheel.sh"
fi

echo ""
log_info "Wheel rebuild complete!"
echo ""
echo "Wheel location: $WHEEL"
echo "Version: $WHEEL_VERSION"
echo ""
if [ "$DEPLOY" = false ]; then
    echo "To deploy to Home Assistant:"
    echo "  $SCRIPT_DIR/deploy-agent-wheel.sh"
fi
