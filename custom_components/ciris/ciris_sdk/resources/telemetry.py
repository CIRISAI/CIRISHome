"""
Telemetry resource for CIRIS v1 API (Pre-Beta).

**WARNING**: This SDK is for the v1 API which is in pre-beta stage.
The API interfaces may change without notice.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

from ..transport import Transport


class TelemetryOverview(BaseModel):
    """System telemetry overview."""
    uptime_seconds: float = Field(..., description="System uptime")
    cognitive_state: str = Field(..., description="Current cognitive state")
    messages_processed_24h: int = Field(default=0, description="Messages in last 24h")
    healthy_services: int = Field(default=0, description="Number of healthy services")
    
    class Config:
        extra = "allow"  # Allow additional fields


class TelemetryMetrics(BaseModel):
    """Telemetry metrics response."""
    metrics: List[Dict[str, Any]] = Field(..., description="List of metrics")
    
    class Config:
        extra = "allow"


class TelemetryMetricDetail(BaseModel):
    """Detailed metric information."""
    metric_name: str = Field(..., description="Metric name")  
    current: float = Field(..., description="Current value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    
    class Config:
        extra = "allow"


class TelemetryResources(BaseModel):
    """Resource telemetry."""
    current: Dict[str, Any] = Field(..., description="Current usage")
    limits: Dict[str, Any] = Field(..., description="Resource limits")
    health: Union[str, Dict[str, Any]] = Field(..., description="Health status")
    
    class Config:
        extra = "allow"


class TelemetryResourcesHistory(BaseModel):
    """Historical resource data."""
    period: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Time period")
    cpu: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(..., description="CPU history")
    memory: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(..., description="Memory history")
    
    class Config:
        extra = "allow"
        
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "TelemetryResourcesHistory":
        """Convert API response to model."""
        # Handle SuccessResponse wrapper
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]
            
        # Extract period from nested data if needed
        period = data.get("period")
        if not period:
            # Create a period string from available data
            if "start" in data and "end" in data:
                period = f"{data['start']} to {data['end']}"
            elif "hours" in data:
                period = f"Last {data['hours']} hours"
            else:
                period = "Recent"
            
        # Extract CPU and memory data
        cpu = data.get("cpu", [])
        if isinstance(cpu, dict):
            if "data" in cpu:
                cpu = cpu["data"]
            else:
                # Keep the whole dict if no data field
                cpu = cpu
            
        memory = data.get("memory", [])
        if isinstance(memory, dict):
            if "data" in memory:
                memory = memory["data"]
            else:
                # Keep the whole dict if no data field
                memory = memory
            
        # If cpu/memory are not present, try history field
        if not cpu and not memory and "history" in data:
            history = data["history"]
            cpu = []
            memory = []
            for entry in history:
                timestamp = entry.get("timestamp")
                if "cpu_percent" in entry:
                    cpu.append({"timestamp": timestamp, "value": entry["cpu_percent"]})
                if "memory_mb" in entry:
                    memory.append({"timestamp": timestamp, "value": entry["memory_mb"]})
            
        return cls(period=period, cpu=cpu, memory=memory)


class TelemetryResource:
    def __init__(self, transport: Transport):
        self._transport = transport

    async def get_overview(self) -> Dict[str, Any]:
        """
        Get system metrics summary.

        Returns comprehensive overview combining telemetry, visibility, incidents, and resource usage.
        """
        data = await self._transport.request("GET", "/v1/telemetry/overview")
        return data

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics.

        Returns detailed metrics with trends and breakdowns by service.
        """
        data = await self._transport.request("GET", "/v1/telemetry/metrics")
        return data

    async def get_traces(
        self,
        limit: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get reasoning traces.

        Returns reasoning traces showing agent thought processes and decision-making.
        """
        params = {"limit": str(limit)}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        data = await self._transport.request("GET", "/v1/telemetry/traces", params=params)
        return data

    async def get_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None,
        service: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get system logs.

        Returns system logs from all services with filtering capabilities.
        """
        params = {"limit": str(limit)}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        if level:
            params["level"] = level
        if service:
            params["service"] = service

        data = await self._transport.request("GET", "/v1/telemetry/logs", params=params)
        return data

    async def query(
        self,
        query_type: str,
        filters: Optional[Dict[str, Any]] = None,
        aggregations: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Execute custom telemetry queries.

        Query types: metrics, traces, logs, incidents, insights
        Requires ADMIN role.
        """
        payload = {
            "query_type": query_type,
            "filters": filters or {},
            "limit": limit
        }

        if aggregations:
            payload["aggregations"] = aggregations
        if start_time:
            payload["start_time"] = start_time.isoformat()
        if end_time:
            payload["end_time"] = end_time.isoformat()

        data = await self._transport.request("POST", "/v1/telemetry/query", json=payload)
        return data

    # Legacy compatibility methods (will be deprecated)
    async def get_observability_overview(self) -> Dict[str, Any]:
        """
        DEPRECATED: Use get_overview() instead.
        Get unified observability overview.
        """
        return await self.get_overview()

    async def get_observability_metrics(self) -> Dict[str, Any]:
        """
        DEPRECATED: Use get_metrics() instead.
        Get detailed system metrics.
        """
        return await self.get_metrics()

    async def get_observability_traces(
        self,
        limit: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use get_traces() instead.
        Get reasoning traces.
        """
        return await self.get_traces(limit=limit, start_time=start_time, end_time=end_time)

    async def get_observability_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None,
        service: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use get_logs() instead.
        Get system logs.
        """
        return await self.get_logs(
            start_time=start_time,
            end_time=end_time,
            level=level,
            service=service,
            limit=limit
        )

    async def query_observability(
        self,
        query_type: str,
        filters: Optional[Dict[str, Any]] = None,
        aggregations: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use query() instead.
        Execute custom observability queries.
        """
        return await self.query(
            query_type=query_type,
            filters=filters,
            aggregations=aggregations,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    # Aliases for backward compatibility with tests
    async def overview(self) -> TelemetryOverview:
        """Alias for get_overview()."""
        data = await self.get_overview()
        return TelemetryOverview(**data)
    
    async def metrics(self) -> TelemetryMetrics:
        """Alias for get_metrics()."""
        data = await self.get_metrics()
        return TelemetryMetrics(**data)
    
    async def metric_detail(self, metric_name: str) -> TelemetryMetricDetail:
        """Get detailed information about a specific metric."""
        data = await self._transport.request("GET", f"/v1/telemetry/metrics/{metric_name}")
        # Handle both direct response and data wrapped response
        if "metric_name" not in data and "name" in data:
            data["metric_name"] = data["name"]
        if "current" not in data and "current_value" in data:
            data["current"] = data["current_value"]
        return TelemetryMetricDetail(**data)
    
    async def resources(self) -> TelemetryResources:
        """Get resource usage telemetry."""
        data = await self._transport.request("GET", "/v1/telemetry/resources")
        return TelemetryResources(**data)
    
    async def resources_history(self, hours: int = 24) -> TelemetryResourcesHistory:
        """Get historical resource usage."""
        params = {"hours": str(hours)}
        data = await self._transport.request("GET", "/v1/telemetry/resources/history", params=params)
        return TelemetryResourcesHistory.from_api_response(data)
