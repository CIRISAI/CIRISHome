#!/bin/bash
# CIRIS Voice Wyoming Bridge Setup Script

echo "CIRIS Voice Wyoming Bridge Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if CIRIS SDK is available
echo "Checking CIRIS SDK..."
if ! pip show ciris-sdk > /dev/null 2>&1; then
    echo "WARNING: CIRIS SDK not found. It may not be published yet."
    echo "You can install it manually when available: pip install ciris-sdk"
fi

# Copy example config if needed
if [ ! -f "config.yaml" ]; then
    echo "Creating config.yaml from example..."
    cp config.example.yaml config.yaml
    echo "Please edit config.yaml with your settings:"
    echo "  - CIRIS API URL and authentication"
    echo "  - STT/TTS provider API keys"
    echo "  - Wyoming server settings"
fi

# Create systemd service file
echo "Creating systemd service file..."
cat > ciris-wyoming.service << EOF
[Unit]
Description=CIRIS Wyoming Voice Bridge
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python -m src.bridge
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your settings"
echo "2. Test the bridge: python -m src.bridge"
echo "3. Install as service: sudo cp ciris-wyoming.service /etc/systemd/system/"
echo "4. Enable service: sudo systemctl enable ciris-wyoming"
echo "5. Start service: sudo systemctl start ciris-wyoming"
echo ""
echo "For Home Assistant:"
echo "- Go to Settings > Voice Assistants"
echo "- Create new pipeline with 60s timeout"
echo "- Select Wyoming protocol"
echo "- Point to this server's IP:10300"
