# CIRIS Agent - Home Assistant Addon

**AI-powered home automation with multi-modal capabilities**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

## Quick Install

### Option 1: Add Repository (Recommended)

1. In Home Assistant, go to **Settings** > **Add-ons** > **Add-on Store**
2. Click the **...** menu (top right) > **Repositories**
3. Add: `https://github.com/CIRISAI/CIRISHome`
4. Find **CIRIS Agent** in the store and click **Install**
5. Start the addon - CIRIS appears in your sidebar automatically

### Option 2: Local Installation

```bash
# SSH into your Home Assistant
ssh root@homeassistant.local

# Create addon directory
mkdir -p /addons/ciris_agent
cd /addons/ciris_agent

# Download latest release
curl -L https://github.com/CIRISAI/CIRISHome/releases/latest/download/ciris-agent-addon.tar.gz | tar xz

# Reload and install
ha addons reload
ha addons install local_ciris_agent
ha addons start local_ciris_agent
```

## First Run Setup

After installation, CIRIS guides you through setup:

1. **Open the CIRIS panel** from the Home Assistant sidebar
2. **Choose your LLM provider**:
   - **OpenRouter** - Multi-model access (recommended)
   - **Anthropic** - Claude API
   - **Local Ollama** - Self-hosted models
   - **Jetson** - Local Llama on Jetson Nano
3. **Enter your API key** (or configure local endpoint)
4. **Complete setup** - CIRIS verifies connectivity

**What happens automatically:**
- CIRIS conversation agent custom component installs to `/config/custom_components/ciris`
- CIRIS integration auto-configures via HA REST API
- HA restarts if needed to load new component (handled automatically)

Just select CIRIS as your voice assistant in **Settings** > **Voice Assistants**!

## Voice Assistant Integration

**Fully Automatic!** After addon startup:

1. **Custom component** auto-installs to Home Assistant
2. **CIRIS integration** auto-configures via REST API
3. Just go to **Settings** > **Voice Assistants** and select CIRIS

If this is a fresh install, HA may restart automatically to load the component. The addon handles everything - no manual steps required.

**Manual Setup (if needed):**

If auto-config didn't run, you can manually add the integration:
1. Go to **Settings** > **Devices & Services** > **Add Integration**
2. Search for **CIRIS** and add it
3. Use API URL: `http://local-ciris_agent:8099`

Now you can use CIRIS with Voice PE pucks, the HA mobile app, or any HA voice interface.

## Features

- **Voice Commands** - Natural language home control
- **Multi-Modal AI** - Vision, audio, and sensor integration
- **100% Local Option** - Use Jetson Nano for complete privacy
- **Home Assistant Native** - Full HA API integration
- **Auto-Install** - Conversation agent installs automatically

## Configuration Options

Configure via the Options tab in the addon:

| Option                 | Description                              | Default        |
| ---------------------- | ---------------------------------------- | -------------- |
| Room Type              | adult_room, kids_room, shared_space, etc | shared_space   |
| Safety Level           | unrestricted, family_friendly, kids_safe | family_friendly|
| Response Style         | normal, simplified, detailed, brief      | normal         |
| Custom Instructions    | Additional context for the AI            | (empty)        |

## Supported Architectures

- `aarch64` (Raspberry Pi 4/5, Home Assistant Yellow)
- `amd64` (Intel/AMD x64)

## Troubleshooting

**View Logs:**

```bash
# Quick log check
ssh root@homeassistant.local 'tail -50 /share/ciris_logs/latest.log'

# Or via HA CLI
ha addons logs local_ciris_agent
```

**Restart Addon:**

```bash
ha addons restart local_ciris_agent
```

**Logs Directory:**

Addon logs are available in `/share/ciris_logs/`:

- `latest.log` - Main application log
- `startup.log` - Startup sequence
- `incidents.log` - Error tracking

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE)

## Links

- [CIRISHome Repository](https://github.com/CIRISAI/CIRISHome)
- [CIRISAgent Core](https://github.com/CIRISAI/CIRISAgent)
- [Report Issues](https://github.com/CIRISAI/CIRISHome/issues)
