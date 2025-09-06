#!/bin/sh
# CIRIS Wyoming Bridge Setup Script for Home Assistant Yellow
# Run this in Terminal & SSH on your HA Yellow

echo "CIRIS Wyoming Bridge Setup for HA Yellow"
echo "========================================"
echo ""

# Configuration
REPO_URL="https://github.com/CIRISAI/CIRISAgent.git"  # Update with your repo URL
BRANCH="main"  # Change if using different branch
ADDON_NAME="ciris-wyoming"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo "${RED}[✗]${NC} $1"
}

# Check if we're in the right place
if [ ! -d "/config" ]; then
    print_error "This script must be run from Home Assistant Terminal & SSH"
    exit 1
fi

# Step 1: Install dependencies
print_status "Installing required packages..."
apk update
apk add --no-cache git

# Step 2: Create addons directory
print_status "Creating local addons directory..."
mkdir -p /addons

# Step 3: Clone or update repository
if [ -d "/addons/$ADDON_NAME" ]; then
    print_warning "Addon directory exists. Updating..."
    cd /addons/$ADDON_NAME

    # Save any local changes
    if [ -f "config.yaml" ]; then
        cp config.yaml config.yaml.backup
        print_status "Backed up existing config to config.yaml.backup"
    fi

    # Re-clone to get latest
    cd /addons
    rm -rf temp 2>/dev/null || true
    print_status "Cloning latest CIRIS repository..."
    git clone --depth 1 $REPO_URL temp || {
        print_error "Failed to clone repository"
        exit 1
    }

    # Copy updated files
    if [ -d "temp/CIRISVoice" ]; then
        cp -r temp/CIRISVoice/src $ADDON_NAME/ 2>/dev/null || print_warning "No src directory found"
        cp temp/CIRISVoice/requirements.txt $ADDON_NAME/ 2>/dev/null || print_warning "No requirements.txt found"
        cp temp/CIRISVoice/config.example.yaml $ADDON_NAME/ 2>/dev/null || print_warning "No config.example.yaml found"
        cp temp/CIRISVoice/docker-addon/* $ADDON_NAME/ 2>/dev/null || print_warning "No docker-addon files found"
        # Copy SDK for local installation
        cp -r temp/ciris_sdk $ADDON_NAME/sdk/ 2>/dev/null || print_warning "No SDK directory found"
    else
        print_error "CIRISVoice directory not found in repository!"
        print_warning "Repository structure:"
        ls -la temp/
    fi

    rm -rf temp
else
    print_status "Creating addon directory..."
    mkdir -p /addons/$ADDON_NAME
    cd /addons

    print_status "Cloning CIRIS repository..."
    git clone --depth 1 $REPO_URL temp || {
        print_error "Failed to clone repository"
        exit 1
    }

    # Copy what we need
    print_status "Setting up addon structure..."

    if [ -d "temp/CIRISVoice" ]; then
        # Copy the Python application files
        cp -r temp/CIRISVoice/src $ADDON_NAME/ 2>/dev/null || print_warning "No src directory found"
        cp temp/CIRISVoice/requirements.txt $ADDON_NAME/ 2>/dev/null || print_warning "No requirements.txt found"
        cp temp/CIRISVoice/config.example.yaml $ADDON_NAME/ 2>/dev/null || print_warning "No config.example.yaml found"

        # Copy the addon Docker files
        cp temp/CIRISVoice/docker-addon/* $ADDON_NAME/ 2>/dev/null || print_warning "No docker-addon files found"

        # Copy SDK for local installation
        cp -r temp/ciris_sdk $ADDON_NAME/sdk/ 2>/dev/null || print_warning "No SDK directory found"

        print_status "Files copied successfully"
    else
        print_error "CIRISVoice directory not found in repository!"
        print_warning "Repository structure:"
        ls -la temp/
        rm -rf temp
        exit 1
    fi

    rm -rf temp
fi

# Step 4: Verify structure
cd /addons/$ADDON_NAME
print_status "Verifying addon structure..."

# Check required addon files
REQUIRED_ADDON_FILES="config.yaml Dockerfile run.sh build.yaml"
MISSING_ADDON_FILES=""

for file in $REQUIRED_ADDON_FILES; do
    if [ ! -f "$file" ]; then
        MISSING_ADDON_FILES="$MISSING_ADDON_FILES $file"
    fi
done

if [ -n "$MISSING_ADDON_FILES" ]; then
    print_error "Missing required addon files:$MISSING_ADDON_FILES"
fi

# Check required app files
REQUIRED_APP_FILES="requirements.txt config.example.yaml"
MISSING_APP_FILES=""

for file in $REQUIRED_APP_FILES; do
    if [ ! -f "$file" ]; then
        MISSING_APP_FILES="$MISSING_APP_FILES $file"
    fi
done

if [ -n "$MISSING_APP_FILES" ]; then
    print_error "Missing required app files:$MISSING_APP_FILES"
fi

# Check src directory
if [ ! -d "src" ]; then
    print_error "Missing src directory!"
else
    print_status "Found src directory with $(find src -name "*.py" | wc -l) Python files"
fi

# Show current structure
print_status "Current addon structure:"
ls -la

# Step 5: Update config with user settings
print_status "Configuring addon..."

# Check if google_cloud_key.json exists
if [ -f "/config/google_cloud_key.json" ]; then
    print_status "Found Google Cloud key at /config/google_cloud_key.json"
else
    print_warning "Google Cloud key not found. You'll need to configure STT/TTS provider."
fi

# Create a setup summary file
cat > /config/ciris_wyoming_setup.txt << EOF
CIRIS Wyoming Bridge Setup Complete!
===================================

Next Steps:

1. Go to Settings → Add-ons → Add-on Store
2. Click the menu (⋮) → Check for updates
3. Look for "CIRIS Wyoming Bridge" in Local add-ons
4. Click on it and:
   - Install the add-on
   - Configure your CIRIS API URL
   - Start the add-on

5. Configure Voice Pipeline:
   - Settings → Voice assistants
   - Add new assistant "CIRIS"
   - Set timeout to 60 seconds
   - Select Wyoming for STT/TTS

6. Assign to Voice PE pucks:
   - Settings → Devices → ESPHome
   - Select your Voice PE device
   - Configure → Select CIRIS pipeline

Configuration needed in addon:
- CIRIS API URL: http://YOUR_CIRIS_IP:8080
- Timeout: 58 seconds
- STT Provider: google (if using Google Cloud)
- TTS Provider: google (if using Google Cloud)

Your Google Cloud key is at: /config/google_cloud_key.json

For help, see: /addons/$ADDON_NAME/VOICE_PE_QUICKSTART.md
EOF

print_status "Setup complete! Check /config/ciris_wyoming_setup.txt for next steps."

# Step 6: Show summary
echo ""
echo "Summary:"
echo "--------"
print_status "Addon location: /addons/$ADDON_NAME"
print_status "Google key: /config/google_cloud_key.json"
print_status "Next: Install addon from HA UI"
echo ""
echo "To update in the future, run this script again!"
