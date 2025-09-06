#!/bin/bash
set -e

echo "CIRIS Wyoming Bridge Installer"
echo "=============================="

python_version=$(python3 --version | cut -d' ' -f2)
echo "Python version: $python_version"

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

if [ ! -f config.yaml ]; then
    echo "Creating config.yaml from example..."
    cp config.yaml.example config.yaml
fi

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your API keys"
echo "2. Set environment variables:"
echo "   export STT_API_KEY='your-openai-or-google-key'"
echo "   export TTS_API_KEY='your-openai-or-google-key'"
echo "3. Start CIRIS agent: python main.py --mode api --profile home_assistant"
echo "4. Start Wyoming bridge: python -m src.bridge"
echo ""
echo "For Home Assistant configuration, add to configuration.yaml:"
echo ""
echo "wyoming:"
echo "  - name: CIRIS"
echo "    uri: tcp://localhost:10300"
