#!/bin/bash
# rebuild-mobile-web.sh
# Rebuilds mobile-web from CIRISAgent/mobile with Kotlin 2.0 compose + wasmJs target
#
# This script ensures the web UI stays in sync with the main mobile codebase
# while adding web-specific targets and configurations.
#
# Usage: ./scripts/rebuild-mobile-web.sh [--clean]
#   --clean: Remove existing mobile-web before rebuilding (default: incremental)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="/home/emoore/CIRISAgent/mobile"
TARGET_DIR="${REPO_ROOT}/mobile-web"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean) CLEAN_BUILD=true ;;
    esac
done

# Verify source exists
if [ ! -d "$SOURCE_DIR" ]; then
    log_error "Source directory not found: $SOURCE_DIR"
    exit 1
fi

log_info "Rebuilding mobile-web from CIRISAgent/mobile"
log_info "Source: $SOURCE_DIR"
log_info "Target: $TARGET_DIR"

# Step 1: Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_step "Cleaning existing mobile-web..."
    rm -rf "$TARGET_DIR"
fi

mkdir -p "$TARGET_DIR"

# Step 2: Copy shared module (the core cross-platform code)
log_step "Copying shared module..."
rm -rf "$TARGET_DIR/shared"
cp -r "$SOURCE_DIR/shared" "$TARGET_DIR/shared"

# Step 3: Copy generated-api module
log_step "Copying generated-api module..."
rm -rf "$TARGET_DIR/generated-api"
cp -r "$SOURCE_DIR/generated-api" "$TARGET_DIR/generated-api"

# Step 4: Copy gradle files
log_step "Copying gradle configuration..."
cp -r "$SOURCE_DIR/gradle" "$TARGET_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/gradlew" "$TARGET_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/gradle.properties" "$TARGET_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/local.properties" "$TARGET_DIR/" 2>/dev/null || true

# Step 5: Copy documentation (excluding platform-specific)
log_step "Copying documentation..."
for doc in README.md CLAUDE.md; do
    [ -f "$SOURCE_DIR/$doc" ] && cp "$SOURCE_DIR/$doc" "$TARGET_DIR/"
done

# Step 6: Create root build.gradle.kts for Kotlin 2.0 + Compose
log_step "Creating root build.gradle.kts with Kotlin 2.0..."
cat > "$TARGET_DIR/build.gradle.kts" << 'EOF'
plugins {
    // Kotlin 2.0 with new Compose compiler architecture
    kotlin("multiplatform").version("2.0.21").apply(false)
    kotlin("plugin.serialization").version("2.0.21").apply(false)
    kotlin("plugin.compose").version("2.0.21").apply(false)
    id("org.jetbrains.compose").version("1.7.1").apply(false)
}
EOF

# Step 7: Create settings.gradle.kts
log_step "Creating settings.gradle.kts..."
cat > "$TARGET_DIR/settings.gradle.kts" << 'EOF'
pluginManagement {
    repositories {
        gradlePluginPortal()
        google()
        mavenCentral()
        maven("https://maven.pkg.jetbrains.space/public/p/compose/dev")
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven("https://maven.pkg.jetbrains.space/public/p/compose/dev")
    }
}

rootProject.name = "ciris-web"
include(":shared")
include(":generated-api")
include(":webApp")
EOF

# Step 8: Update gradle.properties for Kotlin 2.0
log_step "Updating gradle.properties..."
cat > "$TARGET_DIR/gradle.properties" << 'EOF'
kotlin.code.style=official
kotlin.mpp.stability.nowarn=true
kotlin.mpp.enableCInteropCommonization=true
org.jetbrains.compose.experimental.wasm.enabled=true
compose.resources.always.generate.accessors=true
EOF

# Step 9: Update shared/build.gradle.kts for wasmJs target
log_step "Configuring shared module for wasmJs..."
cat > "$TARGET_DIR/shared/build.gradle.kts" << 'EOF'
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform")
    kotlin("plugin.serialization")
    id("org.jetbrains.compose")
    kotlin("plugin.compose")
}

kotlin {
    // Web target (wasmJs)
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        moduleName = "ciris-shared"
        browser {
            commonWebpackConfig {
                outputFileName = "ciris-shared.js"
            }
        }
        binaries.executable()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                // Compose Multiplatform
                implementation(compose.runtime)
                implementation(compose.foundation)
                implementation(compose.material3)
                implementation(compose.materialIconsExtended)
                implementation(compose.components.resources)
                @OptIn(org.jetbrains.compose.ExperimentalComposeLibrary::class)
                implementation(compose.components.uiToolingPreview)

                // Coroutines
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")

                // Serialization
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

                // Date/Time
                implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")

                // Ktor client
                implementation("io.ktor:ktor-client-core:3.0.3")
                implementation("io.ktor:ktor-client-content-negotiation:3.0.3")
                implementation("io.ktor:ktor-serialization-kotlinx-json:3.0.3")
                implementation("io.ktor:ktor-client-logging:3.0.3")
                implementation("io.ktor:ktor-client-auth:3.0.3")

                // Multiplatform ViewModel
                implementation("org.jetbrains.androidx.lifecycle:lifecycle-viewmodel-compose:2.8.4")
                implementation("org.jetbrains.androidx.lifecycle:lifecycle-runtime-compose:2.8.4")

                // Navigation
                implementation("org.jetbrains.androidx.navigation:navigation-compose:2.8.0-alpha10")

                // Generated API client
                implementation(project(":generated-api"))
            }
        }

        val wasmJsMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-js:3.0.3")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
            }
        }
    }
}
EOF

# Step 10: Create wasmJsMain platform implementations
log_step "Creating wasmJsMain platform implementations..."
WASM_MAIN="$TARGET_DIR/shared/src/wasmJsMain/kotlin/ai/ciris/mobile/shared/platform"
mkdir -p "$WASM_MAIN"

# Platform.wasmJs.kt - platform detection
cat > "$WASM_MAIN/Platform.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

import kotlinx.browser.window

actual fun getPlatform(): Platform = Platform.WEB

actual fun platformLog(tag: String, message: String) {
    println("[$tag] $message")
}

actual fun getDeviceDebugInfo(): String {
    return buildString {
        appendLine("Platform: Web (WASM)")
        appendLine("User Agent: ${window.navigator.userAgent}")
        appendLine("Language: ${window.navigator.language}")
    }
}

actual fun openUrlInBrowser(url: String) {
    window.open(url, "_blank")
}

actual fun getAppVersion(): String = "2.3.2"

actual fun getAppBuildNumber(): String = "0"

actual fun startTestAutomationServer() {
    // No-op on web
}
EOF

# SecureStorage.wasmJs.kt - browser localStorage
cat > "$WASM_MAIN/SecureStorage.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

import kotlinx.browser.localStorage

actual class SecureStorage actual constructor() {

    actual suspend fun saveApiKey(key: String, value: String): Result<Unit> = runCatching {
        localStorage.setItem("apikey_$key", value)
    }

    actual suspend fun getApiKey(key: String): Result<String?> = runCatching {
        localStorage.getItem("apikey_$key")
    }

    actual suspend fun saveAccessToken(token: String): Result<Unit> = runCatching {
        localStorage.setItem("ciris_access_token", token)
    }

    actual suspend fun getAccessToken(): Result<String?> = runCatching {
        localStorage.getItem("ciris_access_token")
    }

    actual suspend fun deleteAccessToken(): Result<Unit> = runCatching {
        localStorage.removeItem("ciris_access_token")
    }

    actual suspend fun save(key: String, value: String): Result<Unit> = runCatching {
        localStorage.setItem(key, value)
    }

    actual suspend fun get(key: String): Result<String?> = runCatching {
        localStorage.getItem(key)
    }

    actual suspend fun delete(key: String): Result<Unit> = runCatching {
        localStorage.removeItem(key)
    }

    actual suspend fun clear(): Result<Unit> = runCatching {
        localStorage.clear()
    }
}

actual fun createSecureStorage(): SecureStorage = SecureStorage()
EOF

# PythonRuntime.wasmJs.kt - no-op for web (backend handles Python)
cat > "$WASM_MAIN/PythonRuntime.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

/**
 * Web implementation of PythonRuntime - no-op since backend handles Python.
 * The web UI connects to a remote CIRIS agent via HTTP API.
 */
actual class PythonRuntime actual constructor() {
    private var _initialized = false
    private var _serverStarted = false

    actual suspend fun initialize(pythonHome: String): Result<Unit> = runCatching {
        _initialized = true
    }

    actual suspend fun startServer(): Result<String> = runCatching {
        _serverStarted = true
        serverUrl
    }

    actual suspend fun startPythonServer(onStatus: ((String) -> Unit)?): Result<String> = runCatching {
        onStatus?.invoke("Web mode - connecting to remote server...")
        _serverStarted = true
        serverUrl
    }

    actual fun injectPythonConfig(config: Map<String, String>) {
        // No-op on web - config is on server side
    }

    actual suspend fun checkHealth(): Result<Boolean> = Result.success(true)

    actual suspend fun getServicesStatus(): Result<Pair<Int, Int>> = Result.success(22 to 22)

    actual suspend fun getPrepStatus(): Result<Pair<Int, Int>> = Result.success(2 to 2)

    actual fun shutdown() {
        _serverStarted = false
    }

    actual fun isInitialized(): Boolean = _initialized

    actual fun isServerStarted(): Boolean = _serverStarted

    actual val serverUrl: String = ""  // Empty = relative URLs for ingress
}

actual fun createPythonRuntime(): PythonRuntime = PythonRuntime()
EOF

# EnvFileUpdater.wasmJs.kt - no-op for web
cat > "$WASM_MAIN/EnvFileUpdater.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

import ai.ciris.mobile.shared.config.CIRISConfig

/**
 * Web implementation of EnvFileUpdater - no-op since web doesn't have .env files.
 * Configuration is handled server-side.
 */
actual class EnvFileUpdater {

    actual suspend fun updateEnvWithToken(oauthIdToken: String): Result<Boolean> = Result.success(true)

    actual fun triggerConfigReload() {
        // No-op on web
    }

    actual suspend fun readLlmConfig(): EnvLlmConfig? = null

    actual suspend fun deleteEnvFile(): Result<Boolean> = Result.success(true)

    actual fun checkTokenRefreshSignal(): Boolean = false

    actual suspend fun clearSigningKey(): Result<Boolean> = Result.success(true)

    actual suspend fun clearDataOnly(): Result<Boolean> = Result.success(true)
}

actual fun createEnvFileUpdater(): EnvFileUpdater = EnvFileUpdater()
EOF

# Additional platform stubs needed
# BackHandler.wasmJs.kt
cat > "$WASM_MAIN/BackHandler.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

import androidx.compose.runtime.Composable

@Composable
actual fun PlatformBackHandler(enabled: Boolean, onBack: () -> Unit) {
    // No-op on web - browser handles back button
}
EOF

# AppRestarter.wasmJs.kt
cat > "$WASM_MAIN/AppRestarter.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

import kotlinx.browser.window

actual object AppRestarter {
    actual fun restartApp() {
        window.location.reload()
    }
}
EOF

# Logger.wasmJs.kt
cat > "$WASM_MAIN/Logger.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

actual object PlatformLogger {
    actual fun d(tag: String, message: String) {
        println("[D][$tag] $message")
    }
    actual fun i(tag: String, message: String) {
        println("[I][$tag] $message")
    }
    actual fun w(tag: String, message: String) {
        println("[W][$tag] $message")
    }
    actual fun e(tag: String, message: String) {
        println("[E][$tag] $message")
    }
    actual fun e(tag: String, message: String, throwable: Throwable) {
        println("[E][$tag] $message: ${throwable.message}")
    }
}
EOF

# DebugLogBuffer.wasmJs.kt
cat > "$WASM_MAIN/DebugLogBuffer.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

actual object DebugLogBuffer {
    actual fun append(message: String) {
        console.log(message)
    }
    actual fun getContents(): String = ""
    actual fun clear() {}
}
EOF

# TestAutomation.wasmJs.kt
cat > "$WASM_MAIN/TestAutomation.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

actual object TestAutomation {
    actual fun isEnabled(): Boolean = false
    actual fun setCurrentScreen(screen: String) {}
}
EOF

# KeyboardPadding.wasmJs.kt
cat > "$WASM_MAIN/KeyboardPadding.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.dp

@Composable
actual fun keyboardPadding(): PaddingValues = PaddingValues(0.dp)
EOF

# FilePicker.wasmJs.kt
cat > "$WASM_MAIN/FilePicker.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

import androidx.compose.runtime.Composable

actual class FilePickerLauncher actual constructor(
    private val onResult: (String?) -> Unit
) {
    actual fun launch() {
        // No-op on web - could be implemented with file input element
        onResult(null)
    }
}

@Composable
actual fun rememberFilePickerLauncher(onResult: (String?) -> Unit): FilePickerLauncher {
    return FilePickerLauncher(onResult)
}
EOF

# LocalInferenceCapability.wasmJs.kt
cat > "$WASM_MAIN/LocalInferenceCapability.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

actual object LocalInferenceCapability {
    actual fun isAvailable(): Boolean = false
    actual fun getStatus(): String = "Not available on web"
}
EOF

# LocalLLMServer.wasmJs.kt
cat > "$WASM_MAIN/LocalLLMServer.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

actual object LocalLLMServer {
    actual suspend fun start(): Result<String> = Result.failure(UnsupportedOperationException("Not available on web"))
    actual suspend fun stop(): Result<Unit> = Result.success(Unit)
    actual fun isRunning(): Boolean = false
}
EOF

# KMPFileLogger.wasmJs.kt
cat > "$WASM_MAIN/KMPFileLogger.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

actual object KMPFileLogger {
    actual fun log(level: String, tag: String, message: String) {
        console.log("[$level][$tag] $message")
    }
    actual fun getLogFilePath(): String? = null
}
EOF

# ScheduledTaskNotifications.wasmJs.kt
cat > "$WASM_MAIN/ScheduledTaskNotifications.wasmJs.kt" << 'EOF'
package ai.ciris.mobile.shared.platform

actual object ScheduledTaskNotifications {
    actual fun scheduleNotification(taskId: String, title: String, message: String, triggerTimeMs: Long) {}
    actual fun cancelNotification(taskId: String) {}
    actual fun cancelAllNotifications() {}
}
EOF

# Step 11: Create webApp module
log_step "Creating webApp module..."
mkdir -p "$TARGET_DIR/webApp/src/wasmJsMain/kotlin/ai/ciris/web"
mkdir -p "$TARGET_DIR/webApp/src/wasmJsMain/resources"

# webApp/build.gradle.kts
cat > "$TARGET_DIR/webApp/build.gradle.kts" << 'EOF'
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform")
    id("org.jetbrains.compose")
    kotlin("plugin.compose")
}

kotlin {
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        moduleName = "ciris-web"
        browser {
            commonWebpackConfig {
                outputFileName = "ciris-web.js"
            }
        }
        binaries.executable()
    }

    sourceSets {
        val wasmJsMain by getting {
            dependencies {
                implementation(project(":shared"))
                implementation(compose.runtime)
                implementation(compose.foundation)
                implementation(compose.material3)
            }
        }
    }
}

compose.resources {
    publicResClass = true
    packageOfResClass = "ai.ciris.web.resources"
    generateResClass = auto
}
EOF

# webApp Main.kt - proper entry point using shared CIRISApp
cat > "$TARGET_DIR/webApp/src/wasmJsMain/kotlin/ai/ciris/web/Main.kt" << 'EOF'
package ai.ciris.web

import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.window.ComposeViewport
import kotlinx.browser.document
import kotlinx.browser.localStorage
import kotlinx.browser.window
import ai.ciris.mobile.shared.CIRISApp
import ai.ciris.mobile.shared.platform.createPythonRuntime
import ai.ciris.mobile.shared.platform.createSecureStorage
import ai.ciris.mobile.shared.platform.createEnvFileUpdater

/**
 * CIRIS Web Application Entry Point
 *
 * This uses the shared CIRISApp from the mobile codebase, providing
 * the same GUI wizard and functionality as Android/iOS/Desktop.
 *
 * For Home Assistant addon mode:
 * - Detects HA ingress context automatically
 * - Skips user creation (HA handles auth)
 * - Uses SUPERVISOR_TOKEN for API calls
 */
@OptIn(ExperimentalComposeUiApi::class)
fun main() {
    val body = document.body ?: return

    // Detect Home Assistant addon mode
    val isHAAddon = detectHAMode()
    if (isHAAddon) {
        // Store HA mode flag for the app to use
        localStorage.setItem("ciris_ha_addon_mode", "true")
    }

    // Get base URL - in HA mode, use relative paths
    val baseUrl = if (isHAAddon) {
        "" // Relative to ingress URL
    } else {
        localStorage.getItem("ciris_base_url")
            ?: getUrlParameter("baseUrl")
            ?: "http://127.0.0.1:8080"
    }

    // Get access token if available
    val accessToken = localStorage.getItem("ciris_access_token")

    ComposeViewport(body) {
        CIRISApp(
            accessToken = accessToken ?: "",
            baseUrl = baseUrl,
            pythonRuntime = createPythonRuntime(),
            secureStorage = createSecureStorage(),
            envFileUpdater = createEnvFileUpdater(),
            isHAAddonMode = isHAAddon
        )
    }
}

/**
 * Detect if running inside Home Assistant addon ingress
 */
private fun detectHAMode(): Boolean {
    val path = window.location.pathname
    return path.contains("/api/hassio_ingress/") ||
           path.contains("/hassio/ingress/") ||
           window.parent != window // Embedded in iframe
}

/**
 * Get URL parameter from browser location
 */
private fun getUrlParameter(name: String): String? {
    val params = window.location.search
    if (params.isEmpty()) return null

    return params.substring(1).split("&")
        .map { it.split("=") }
        .find { it.size == 2 && it[0] == name }
        ?.get(1)
        ?.let { decodeURIComponent(it) }
}

private external fun decodeURIComponent(encodedURI: String): String
EOF

# webApp index.html
cat > "$TARGET_DIR/webApp/src/wasmJsMain/resources/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIRIS</title>
    <style>
        html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            background-color: #0A0A0F;
            overflow: hidden;
        }
        #loading {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #0A0A0F;
            color: #6B9AFF;
            font-family: system-ui, -apple-system, sans-serif;
        }
        #loading h1 { font-size: 48px; margin-bottom: 16px; }
        #loading p { color: rgba(255,255,255,0.6); }
        .spinner {
            width: 40px; height: 40px;
            border: 3px solid #1A1A24;
            border-top-color: #6B9AFF;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-top: 24px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="loading">
        <h1>CIRIS</h1>
        <p>Loading...</p>
        <div class="spinner"></div>
    </div>
    <script src="ciris-web.js"></script>
    <script>
        window.addEventListener('load', function() {
            setTimeout(function() {
                var loading = document.getElementById('loading');
                if (loading) loading.style.display = 'none';
            }, 500);
        });
    </script>
</body>
</html>
EOF

# Step 12: Update generated-api for wasmJs
log_step "Configuring generated-api for wasmJs..."
cat > "$TARGET_DIR/generated-api/build.gradle.kts" << 'EOF'
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform")
    kotlin("plugin.serialization")
}

kotlin {
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
                implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")
                implementation("io.ktor:ktor-client-core:3.0.3")
                implementation("io.ktor:ktor-client-content-negotiation:3.0.3")
                implementation("io.ktor:ktor-serialization-kotlinx-json:3.0.3")
            }
        }
        val wasmJsMain by getting
    }
}
EOF

# Step 13: Post-processing for wasmJs compatibility
log_step "Post-processing for wasmJs compatibility..."

# 13a. Replace Dispatchers.IO with Dispatchers.Default (IO not available on wasmJs)
# Remove the direct import of IO dispatcher extension
find "$TARGET_DIR/shared/src/commonMain" -name "*.kt" -exec sed -i '/import kotlinx\.coroutines\.IO$/d' {} \;
# Replace Dispatchers.IO with Dispatchers.Default
find "$TARGET_DIR/shared/src/commonMain" -name "*.kt" -exec sed -i 's/Dispatchers\.IO/Dispatchers.Default/g' {} \;
# Replace standalone IO (from extension import) with Dispatchers.Default
find "$TARGET_DIR/shared/src/commonMain" -name "*.kt" -exec sed -i 's/withContext(IO)/withContext(Dispatchers.Default)/g' {} \;

# 13b. Remove wasmJsMain files that conflict with actual implementations in commonMain
# These files already have actual implementations defined in commonMain (not expect/actual pattern)
rm -f "$WASM_MAIN/DebugLogBuffer.wasmJs.kt"
rm -f "$WASM_MAIN/KMPFileLogger.wasmJs.kt"
rm -f "$WASM_MAIN/LocalInferenceCapability.wasmJs.kt"
rm -f "$WASM_MAIN/LocalLLMServer.wasmJs.kt"
rm -f "$WASM_MAIN/TestAutomation.wasmJs.kt"
rm -f "$WASM_MAIN/ScheduledTaskNotifications.wasmJs.kt"
rm -f "$WASM_MAIN/FilePicker.wasmJs.kt"
rm -f "$WASM_MAIN/KeyboardPadding.wasmJs.kt"

# 13c. Fix when expressions to add WEB branches for Platform enum
# Add WEB branches to all Platform when expressions
# Handle both short form (Platform.DESKTOP) and fully qualified form (ai.ciris.mobile.shared.platform.Platform.DESKTOP)
find "$TARGET_DIR/shared/src/commonMain" -name "*.kt" -exec sed -i 's/Platform\.DESKTOP -> \([^}]*\)}/Platform.DESKTOP -> \1\n                Platform.WEB -> \1}/g' {} \;
# Handle fully qualified Platform references
find "$TARGET_DIR/shared/src/commonMain" -name "*.kt" -exec sed -i 's/ai\.ciris\.mobile\.shared\.platform\.Platform\.DESKTOP -> \(Platform\.[A-Z]*\)$/ai.ciris.mobile.shared.platform.Platform.DESKTOP -> \1\n                    ai.ciris.mobile.shared.platform.Platform.WEB -> Platform.WEB/g' {} \;

# Step 14: Create build script
log_step "Creating build script..."
cat > "$TARGET_DIR/build-web.sh" << 'EOF'
#!/bin/bash
# Build the CIRIS web app
set -e
cd "$(dirname "$0")"

echo "Building CIRIS Web App..."
./gradlew :webApp:wasmJsBrowserDevelopmentExecutableDistribution

echo ""
echo "Build complete!"
echo "Output: webApp/build/dist/wasmJs/developmentExecutable/"
EOF
chmod +x "$TARGET_DIR/build-web.sh"

log_info "Rebuild complete!"
echo ""
echo "Next steps:"
echo "  1. cd $TARGET_DIR"
echo "  2. ./build-web.sh"
echo "  3. Output will be in webApp/build/dist/wasmJs/developmentExecutable/"
echo ""
echo "To deploy to HA addon:"
echo "  ./scripts/deploy-addon.sh <HA_HOST>"
