# CIRIS Agent - Home Assistant Addon

This directory contains the Home Assistant addon for CIRIS Agent, providing multi-modal AI home automation via the HA Supervisor system.

## Why This Fork?

The upstream CIRIS Agent uses **Compose Multiplatform 1.X** to maintain Android compatibility via Chaquopy bindings. We needed Compose 2.X, so we:

1. **Upgraded Compose Multiplatform to 2.X** - The UI framework itself
2. **Generated the Web UI** - Compiled Compose 2.X to web target, producing `www/`
3. **Pure Python backend** - HA addons require Alpine Linux with `py3-none-any` wheels

### Compose Multiplatform Version Constraints

| Project | Compose Version | Web Target |
|---------|-----------------|------------|
| Upstream CIRIS Agent | 1.X | Not available (Chaquopy/Android bound) |
| This Fork | 2.X | `www/` - compiled web output |

The `www/` directory contains the **generated output** from Compose Multiplatform 2.X targeting web (Kotlin/JS or Kotlin/Wasm).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Home Assistant Supervisor                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CIRIS Agent Addon Container                 │    │
│  │  ┌─────────────────────────────────────────────────────┐│    │
│  │  │  run.sh (#!/usr/bin/with-contenv bashio)           ││    │
│  │  │  - Receives SUPERVISOR_TOKEN from HA               ││    │
│  │  │  - Sets CIRIS_HOME=/data/ciris                     ││    │
│  │  │  - Launches ciris-agent on port 8099               ││    │
│  │  └─────────────────────────────────────────────────────┘│    │
│  │                          ↓                               │    │
│  │  ┌─────────────────────────────────────────────────────┐│    │
│  │  │  ciris-agent (FastAPI)                             ││    │
│  │  │  - /v1/setup/* - Configuration wizard              ││    │
│  │  │  - /v1/chat/* - Chat interface                     ││    │
│  │  │  - gui_static/ - Web UI files                      ││    │
│  │  └─────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓ ingress                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Home Assistant Frontend (sidebar panel "CIRIS")        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Addon manifest - slug, ports, permissions, ingress settings |
| `Dockerfile` | Alpine container build - installs wheel, copies www to gui_static |
| `run.sh` | Startup script - sets env vars, launches agent |
| `build.yaml` | HA build config - base image selection |
| `www/` | Placeholder - actual content copied from Compose build at deploy time |

### Web UI Source

The web UI comes from the Compose Multiplatform build (Kotlin 2.0 + wasmJs):
```
mobile-web/webApp/build/dist/wasmJs/developmentExecutable/
├── index.html
├── ciris-web.js
├── ciris-web.wasm
└── composeResources/
```

**Rebuild from CIRISAgent/mobile:**
```bash
./scripts/rebuild-mobile-web.sh --clean  # Full rebuild
./scripts/rebuild-mobile-web.sh          # Incremental sync
```

**Build web output:**
```bash
cd mobile-web && ./build-web.sh
```

**Deploy to HA:**
```bash
./scripts/deploy-addon.sh <HA_HOST>
```

See `mobile-web/CLAUDE.md` for detailed rebuild instructions.

---

## Critical Configuration

### config.yaml Settings

```yaml
# REQUIRED for SUPERVISOR_TOKEN
hassio_api: true
hassio_role: default

# REQUIRED for HA API access
homeassistant_api: true
auth_api: true

# Ingress configuration
ingress: true
ingress_port: 8099
ingress_panel: true    # Shows in sidebar (but must also enable via API)
panel_icon: mdi:robot
panel_title: CIRIS
```

### run.sh Shebang

```bash
#!/usr/bin/with-contenv bashio
```

**CRITICAL**: Must use `/usr/bin/with-contenv`, NOT `/command/with-contenv`. This injects the `SUPERVISOR_TOKEN` environment variable from HA Supervisor.

### CIRIS_HOME Path

```bash
export CIRIS_HOME=/data/ciris
```

**CRITICAL**: Must use `/data/ciris`, NOT `/root/ciris`. The `/root` directory is forbidden in HA addons. The `/data` directory is persistent addon storage.

---

## Deployment

### Quick Deploy

```bash
# Standard update (uses Docker cache)
./scripts/deploy-addon.sh 192.168.50.243

# Fresh install (forces complete rebuild)
./scripts/deploy-addon.sh 192.168.50.243 --fresh
```

### Deploy Script Steps

1. SSH to HA host
2. Create `/addons/ciris-agent` directory
3. Copy Dockerfile, build.yaml, www/
4. Generate config.yaml with correct settings
5. Generate run.sh with correct shebang
6. Reload addon store
7. Install/rebuild addon
8. Enable sidebar panel via Supervisor API
9. Verify addon is running

### Wheel Deployment

The agent wheel must be deployed separately:

```bash
./scripts/deploy-agent-wheel.sh 192.168.50.243
```

The wheel MUST be pure Python (`py3-none-any`). Alpine uses musllinux, so `manylinux` wheels will fail to install.

---

## Lessons Learned

### Addon Slug Naming
- Local addon slug uses hyphen: `ciris-agent`
- Directory name doesn't have to match slug
- Supervisor discovers local addons as `local_<slug>`

### SUPERVISOR_TOKEN Injection
- Requires `hassio_api: true` in config.yaml
- Also set `hassio_role: default` for proper API access
- Token is injected by Supervisor into container environment
- Use `#!/usr/bin/with-contenv bashio` shebang (NOT `/command/with-contenv`)

### Docker Caching Issues
- `ha addons rebuild` uses Docker layer cache
- Changes to `run.sh` may NOT be picked up by rebuild
- Use `--fresh` flag to force complete uninstall/reinstall
- Bump version in config.yaml to force rebuild

### Sidebar Panel Auto-Enable
- The addon self-enables the sidebar panel on startup via `run.sh`
- Uses Supervisor API: `POST http://supervisor/addons/self/options`
- No manual intervention required - CIRIS appears in sidebar automatically

### Internal Networking
- Use internal hostname for API calls: `local-ciris-agent:8099`
- `localhost:8099` returns 404 from HA's perspective
- Web UI should use relative paths: `v1/setup/verify-status`

---

## Troubleshooting

### Log File Locations (on HA)

Logs are exposed in `/share/ciris_logs/` for easy access:

| File | Description |
|------|-------------|
| `/share/ciris_logs/latest.log` | Main application log (all messages) |
| `/share/ciris_logs/incidents.log` | Incident/error log |
| `/share/ciris_logs/startup.log` | Startup sequence log |
| `/share/ciris_logs/env.txt` | Current environment snapshot |

**Quick log access:**
```bash
# View latest log
ssh root@192.168.50.243 'tail -100 /share/ciris_logs/latest.log'

# Search for errors
ssh root@192.168.50.243 'grep -i error /share/ciris_logs/latest.log | tail -20'

# View incidents
ssh root@192.168.50.243 'cat /share/ciris_logs/incidents.log'
```

### Check Addon Logs (stdout)
```bash
ssh root@192.168.50.243 'ha addons logs local_ciris-agent'
```

### Check Supervisor Logs
```bash
ssh root@192.168.50.243 'ha supervisor logs | grep ciris'
```

### Verify SUPERVISOR_TOKEN
Look for in addon logs:
```
[CIRIS STARTUP] HA Addon Mode: yes (SUPERVISOR_TOKEN: detected)
```

### Verify CIRIS_HOME
Should show:
```
CIRIS_HOME: /data/ciris
```

NOT `/root/ciris` (forbidden system directory).

### Force Complete Rebuild
```bash
./scripts/deploy-addon.sh 192.168.50.243 --fresh
```

### Verify Addon Discovery
```bash
ssh root@192.168.50.243 'ha addons info local_ciris-agent'
```

### Check Ingress Panel
```bash
ssh root@192.168.50.243 'ha addons info local_ciris-agent | grep ingress'
```

Should show `ingress_panel: true`.

---

## Web UI

The web UI is built with Compose Multiplatform 2.X (Kotlin/WASM) and provides:

1. **First Run Wizard** - Initial setup and LLM provider configuration
2. **Chat Interface** - Interact with CIRIS for home automation
3. **Device Control** - Home Assistant device management

### LLM Providers (configured in wizard)

| Provider | Description |
|----------|-------------|
| OpenRouter | Multi-model access (Qwen, Llama, Claude, etc.) |
| Anthropic | Claude API direct |
| Jetson | Local Llama-4-Scout on Jetson Nano |
| Ollama | Local Ollama instance |

---

## Known Issues

### "Web Sign-In not available" on Phone Browser

**Symptom**: OAuth sign-in button shows error on phone browser but works on desktop.

**Root Cause**: The WASM build cannot do native OAuth (`googleSignInCallback = null`). Desktop works because HA addon mode is detected via ingress URL pattern, which skips OAuth entirely and uses BYOK mode.

**Why Phone Fails**:
1. Phone may access HA via different URL (Nabu Casa, companion app, reverse proxy)
2. URL doesn't match `/api/hassio_ingress/` pattern
3. `detectHAMode()` returns false → Login screen shown
4. OAuth button fails because WASM has no OAuth callback

**Workarounds**:
1. Access HA directly via local IP (not Nabu Casa): `http://192.168.x.x:8123/`
2. Or add `?byok=true` to URL to force BYOK mode (if implemented)

**Upstream Fix**: See `docs/UPSTREAM_FIX_WEB_SIGNIN_FLOW.md`

### Kotlin WASM Production Builds Broken (KT-69154)

**Symptom**: Production WASM builds fail at runtime with:
```
WebAssembly.instantiate(): Import #1 'js_code' 'kotlin.wasm.internal.throwJsError':
function import requires a callable
```

**Root Cause**: Kotlin bug KT-69154 - JS wrapper gets out of sync with WASM module during production optimization/minification.

**Workaround**: Use development builds ONLY:
```bash
# Use this:
./gradlew :webApp:wasmJsBrowserDevelopmentExecutable

# NOT this (broken):
./gradlew :webApp:wasmJsBrowserDistribution
```

**Impact**: Development builds are larger (~11MB gzipped vs ~3MB production), but functional.

**Status**: Track upstream fix at https://youtrack.jetbrains.com/issue/KT-69154

**Deploy Script**: `scripts/deploy-addon.sh` is configured to ONLY use development builds.

---

## Version History

| Version | Changes |
|---------|---------|
| 6.0.5 | Fixed CIRIS_HOME path, sidebar panel persistence, API routing |
| 6.0.4 | Added gui_static copying in Dockerfile |
| 6.0.3 | Fixed SUPERVISOR_TOKEN injection |
| 6.0.0 | Initial HA addon structure |
