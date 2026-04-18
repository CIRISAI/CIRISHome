#!/bin/bash
# deploy-agent-wheel.sh
# Builds and deploys CIRISAgent wheel to Home Assistant addon directory
#
# Usage: ./scripts/deploy-agent-wheel.sh [HA_HOST]
#
# Environment variables:
#   CIRIS_AGENT_DIR - Path to CIRISAgent directory (default: ../CIRISAgent)
#
# =============================================================================
# LESSONS LEARNED:
# =============================================================================
#
# 1. WHEEL PLATFORM COMPATIBILITY:
#    - Must build pure Python wheel (py3-none-any), NOT manylinux
#    - HA uses Alpine Linux which uses musllinux, not glibc
#    - manylinux wheels (built on glibc systems) won't install on Alpine
#    - Error: "is not a supported wheel on this platform"
#
# 2. PLATFORM-SPECIFIC FILES:
#    - Any .jar, .so, .dll, or binary files will cause manylinux wheel
#    - Must temporarily remove these files before building
#    - The CIRIS desktop app JAR file is a common culprit
#
# 3. BUILD PROCESS:
#    - Use python3 -m build --wheel for reproducible builds
#    - Clean dist/, build/, *.egg-info before building
#    - Verify wheel filename contains "py3-none-any"
#
# 4. DEPLOYMENT:
#    - Remove old wheels before copying new ones
#    - Wheel goes to /addons/ciris_agent/ on HA
#    - After deploying wheel, run deploy-addon.sh --fresh to rebuild
#
# =============================================================================

set -e

# Configuration
HA_HOST="${1:-192.168.50.243}"
HA_USER="root"
CIRIS_AGENT_DIR="${CIRIS_AGENT_DIR:-../CIRISAgent}"
ADDON_PATH="/addons/ciris_agent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Resolve relative path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$CIRIS_AGENT_DIR" == ../* ]]; then
    CIRIS_AGENT_DIR="$(cd "$SCRIPT_DIR" && cd "$CIRIS_AGENT_DIR" 2>/dev/null && pwd)" || true
fi

# Check prerequisites
if [ ! -d "$CIRIS_AGENT_DIR" ]; then
    log_error "CIRISAgent directory not found at: $CIRIS_AGENT_DIR"
    log_info "Set CIRIS_AGENT_DIR environment variable or ensure ../CIRISAgent exists relative to CIRISHome"
    exit 1
fi

# Check for python build module
if ! python3 -c "import build" 2>/dev/null; then
    log_error "Python 'build' module not found. Install with: pip install build"
    exit 1
fi

cd "$CIRIS_AGENT_DIR"
log_info "Working in: $(pwd)"

# Check SSH connectivity
log_step "Checking SSH connectivity to $HA_HOST..."
if ! ssh -o ConnectTimeout=5 "${HA_USER}@${HA_HOST}" "echo 'SSH OK'" &>/dev/null; then
    log_error "Cannot connect to ${HA_USER}@${HA_HOST}"
    exit 1
fi

# List of platform-specific files that would create manylinux wheel
PLATFORM_FILES=(
    "ciris_engine/desktop_app/CIRIS-linux-x64-2.0.0.jar"
    # Add other platform-specific files here as needed
)

# Move platform-specific files temporarily
MOVED_FILES=()
for file in "${PLATFORM_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_warn "Found platform-specific file: $file"
        log_info "Moving to /tmp temporarily..."
        mv "$file" "/tmp/$(basename $file)"
        MOVED_FILES+=("$file")
    fi
done

# Clean previous builds
log_step "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build pure Python wheel
log_step "Building pure Python wheel..."
python3 -m build --wheel

# Restore moved files
for file in "${MOVED_FILES[@]}"; do
    log_info "Restoring: $file"
    mv "/tmp/$(basename $file)" "$file"
done

# Find the built wheel
WHEEL_FILE=$(ls dist/ciris_agent-*-py3-none-any.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    log_error "No pure Python wheel found!"
    log_error "The build may have created a platform-specific wheel instead."
    echo ""
    echo "Files in dist/:"
    ls -la dist/
    echo ""
    log_info "Check for platform-specific files in the source tree:"
    find . -name "*.so" -o -name "*.jar" -o -name "*.dll" 2>/dev/null | head -10
    exit 1
fi

log_info "Built wheel: $WHEEL_FILE"
WHEEL_SIZE=$(du -h "$WHEEL_FILE" | cut -f1)
log_info "Wheel size: $WHEEL_SIZE"

# Extract version from wheel filename
WHEEL_VERSION=$(basename "$WHEEL_FILE" | sed 's/ciris_agent-\([^-]*\)-.*/\1/')
log_info "Wheel version: $WHEEL_VERSION"

# Verify it's a pure Python wheel
if [[ "$WHEEL_FILE" != *"py3-none-any"* ]]; then
    log_error "Wheel is not pure Python (py3-none-any)."
    log_error "HA uses Alpine (musllinux) and cannot install manylinux wheels."
    exit 1
fi

# Deploy to HA
log_step "Deploying wheel to HA at $HA_HOST..."
ssh "${HA_USER}@${HA_HOST}" "mkdir -p ${ADDON_PATH}"

# Remove old wheels first
log_info "Removing old wheels..."
ssh "${HA_USER}@${HA_HOST}" "rm -f ${ADDON_PATH}/ciris_agent-*.whl"

# Copy new wheel
log_info "Copying wheel to HA..."
scp -q "$WHEEL_FILE" "${HA_USER}@${HA_HOST}:${ADDON_PATH}/"

# Verify deployment
DEPLOYED_WHEEL=$(ssh "${HA_USER}@${HA_HOST}" "ls ${ADDON_PATH}/ciris_agent-*.whl 2>/dev/null" || echo "")
if [ -z "$DEPLOYED_WHEEL" ]; then
    log_error "Wheel deployment failed!"
    exit 1
fi

echo ""
log_info "Wheel deployed successfully!"
echo ""
echo "  Location: ${HA_HOST}:${ADDON_PATH}/$(basename $WHEEL_FILE)"
echo "  Version:  $WHEEL_VERSION"
echo "  Size:     $WHEEL_SIZE"
echo ""
log_info "Next steps:"
echo "  1. Deploy addon:  ./scripts/deploy-addon.sh $HA_HOST --fresh"
echo "  2. Or rebuild:    ssh ${HA_USER}@${HA_HOST} 'ha addons rebuild local_ciris_agent'"
echo ""
log_warn "Note: Use --fresh flag with deploy-addon.sh to ensure Docker picks up the new wheel"
