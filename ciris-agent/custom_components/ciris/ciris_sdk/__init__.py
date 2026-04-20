"""
CIRIS SDK for v1 API (Pre-Beta).

**WARNING**: This SDK is for the v1 API which is in pre-beta stage.
The API and SDK interfaces may change without notice.
No backwards compatibility is guaranteed.
"""

from .client import CIRISClient
from .models import (  # Legacy models; Telemetry models; Other models
    AdapterInfo,
    AuditEntriesResponse,
    AuditEntryResponse,
    AuditExportResponse,
    DeferralInfo,
    MemoryEntry,
    MemoryOpResult,
    MemoryScope,
    Message,
    MetricRecord,
    ProcessorControlResponse,
    ProcessorState,
    RuntimeStatus,
    ServiceInfo,
    SystemHealth,
    TelemetryDetailedMetric,
    TelemetryLogEntry,
    TelemetryMetricData,
    TelemetryReasoningTrace,
    TelemetrySystemOverview,
)
from .resources.agent import (
    AgentIdentity,
    AgentStatus,
    ConversationHistory,
    ConversationMessage,
    InteractResponse,
)
from .resources.emergency import (
    EmergencyCommandType,
    EmergencyShutdownResponse,
    WASignedCommand,
)
from .resources.memory import (
    GraphNode,
    MemoryQueryResponse,
    MemoryStoreResponse,
    TimelineResponse,
)
from .resources.system import (
    ResourceUsageResponse,
    RuntimeControlResponse,
    ServicesStatusResponse,
    ShutdownResponse,
    SystemHealthResponse,
    SystemTimeResponse,
)
from .websocket import EventChannel, WebSocketClient

__all__ = [
    "CIRISClient",
    "WebSocketClient",
    "EventChannel",
    # Agent interaction types
    "InteractResponse",
    "AgentStatus",
    "AgentIdentity",
    "ConversationHistory",
    "ConversationMessage",
    # Memory types
    "GraphNode",
    "MemoryStoreResponse",
    "MemoryQueryResponse",
    "TimelineResponse",
    # System types
    "SystemHealthResponse",
    "SystemTimeResponse",
    "ResourceUsageResponse",
    "RuntimeControlResponse",
    "ServicesStatusResponse",
    "ShutdownResponse",
    # Emergency types
    "EmergencyShutdownResponse",
    "WASignedCommand",
    "EmergencyCommandType",
    # Legacy models
    "MemoryEntry",
    "MemoryScope",
    "MemoryOpResult",
    # Telemetry
    "TelemetryMetricData",
    "TelemetryDetailedMetric",
    "TelemetrySystemOverview",
    "TelemetryReasoningTrace",
    "TelemetryLogEntry",
    # Other models
    "Message",
    "ProcessorControlResponse",
    "AdapterInfo",
    "RuntimeStatus",
    "SystemHealth",
    "ServiceInfo",
    "ProcessorState",
    "MetricRecord",
    "DeferralInfo",
    "AuditEntryResponse",
    "AuditEntriesResponse",
    "AuditExportResponse",
]

# Version indicator for v1 API
__version__ = "1.0.0-pre-beta"
