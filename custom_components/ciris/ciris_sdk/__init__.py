"""
CIRIS SDK for v1 API (Pre-Beta).

**WARNING**: This SDK is for the v1 API which is in pre-beta stage.
The API and SDK interfaces may change without notice.
No backwards compatibility is guaranteed.
"""

from .client import CIRISClient
from .websocket import WebSocketClient, EventChannel
from .resources.agent import (
    InteractResponse,
    AgentStatus,
    AgentIdentity,
    ConversationHistory,
    ConversationMessage
)
from .resources.memory import (
    GraphNode,
    MemoryStoreResponse,
    MemoryQueryResponse,
    TimelineResponse
)
from .resources.system import (
    SystemHealthResponse,
    SystemTimeResponse,
    ResourceUsageResponse,
    RuntimeControlResponse,
    ServicesStatusResponse,
    ShutdownResponse
)
from .resources.emergency import (
    EmergencyShutdownResponse,
    WASignedCommand,
    EmergencyCommandType
)
from .models import (
    # Legacy models
    MemoryEntry,
    MemoryScope,
    MemoryOpResult,
    # Telemetry models
    TelemetryMetricData,
    TelemetryDetailedMetric,
    TelemetrySystemOverview,
    TelemetryReasoningTrace,
    TelemetryLogEntry,
    # Other models
    Message,
    ProcessorControlResponse,
    AdapterInfo,
    RuntimeStatus,
    SystemHealth,
    ServiceInfo,
    ProcessorState,
    MetricRecord,
    DeferralInfo,
    AuditEntryResponse,
    AuditEntriesResponse,
    AuditExportResponse
)

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
    "AuditExportResponse"
]

# Version indicator for v1 API
__version__ = "1.0.0-pre-beta"
