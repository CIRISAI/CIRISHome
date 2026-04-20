# CIRIS Agent - Home Assistant Addon

**AI-powered home automation with multi-modal capabilities**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

## Quick Install

### Option 1: Add Repository (Recommended)

1. In Home Assistant, go to **Settings** > **Add-ons** > **Add-on Store**
2. Click the **⋮** menu (top right) > **Repositories**
3. Add: `https://github.com/CIRISAI/CIRISHome`
4. Find **CIRIS Agent** in the store and click **Install**
5. Start the addon and open the web UI from the sidebar

### Option 2: Local Installation

```bash
# SSH into your Home Assistant
ssh root@homeassistant.local

# Create addon directory
mkdir -p /addons/ciris_agent
cd /addons/ciris_agent

# Download latest release
curl -L https://github.com/CIRISAI/CIRISHome/releases/latest/download/ciris-agent-addon.tar.gz | tar xz

# Reload addon store
ha addons reload

# Install from local addons
ha addons install local_ciris_agent
ha addons start local_ciris_agent
```

## First Run Setup

After installation, CIRIS will guide you through setup:

1. **Open the CIRIS panel** from the Home Assistant sidebar
2. **Choose your LLM provider**:
   - **OpenRouter** - Multi-model access (recommended)
   - **Anthropic** - Claude API
   - **Local Ollama** - Self-hosted models
   - **Jetson** - Local Llama on Jetson Nano
3. **Enter your API key** (or configure local endpoint)
4. **Complete setup** - CIRIS will verify connectivity

## Features

- **Voice Commands** - Natural language home control
- **Multi-Modal AI** - Vision, audio, and sensor integration
- **100% Local Option** - Use Jetson Nano for complete privacy
- **Home Assistant Native** - Full HA API integration
- **Conversation Agent** - Use CIRIS as your HA voice assistant

## Conversation Agent Integration

After setup, enable CIRIS as your Home Assistant conversation agent:

1. Go to **Settings** > **Devices & Services** > **Add Integration**
2. Search for **CIRIS** and add it
3. Go to **Settings** > **Voice Assistants**
4. Set CIRIS as your conversation agent

Now you can use CIRIS with Voice PE pucks, the HA mobile app, or any HA voice interface.

## Configuration

| Option          | Description                           |
| --------------- | ------------------------------------- |
| `ingress_panel` | Show CIRIS in sidebar (default: true) |

All configuration is done through the web UI wizard. No manual YAML editing required.

## Supported Architectures

- `aarch64` (Raspberry Pi 4/5, Home Assistant Yellow)
- `amd64` (Intel/AMD x64)

## Troubleshooting

**View Logs:**

```bash
ha addons logs local_ciris_agent
```

**Restart Addon:**

```bash
ha addons restart local_ciris_agent
```

**Check Status:**

```bash
ha addons info local_ciris_agent
```

**Logs Directory:**
Addon logs are copied to `/share/ciris_logs/` for easy access:

- `latest.log` - Main application log
- `startup.log` - Startup sequence
- `env.txt` - Current environment (API keys redacted)

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE)

## Links

- [CIRISHome Repository](https://github.com/CIRISAI/CIRISHome)
- [CIRISAgent Core](https://github.com/CIRISAI/CIRISAgent)
- [Report Issues](https://github.com/CIRISAI/CIRISHome/issues)
