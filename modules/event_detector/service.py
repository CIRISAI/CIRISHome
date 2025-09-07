"""
Local Event Detection Service.

Replaces Google Nest cloud event detection with 100% local processing using
Llama-4-Scout vision capabilities and Jetson GPU acceleration.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    """Represents a detected event."""

    event_type: str
    camera_name: str
    confidence: float
    timestamp: datetime
    zones: List[str]
    description: str
    image_data: Optional[bytes] = None


class LocalEventDetectionService:
    """Local event detection service using Llama-4-Scout vision."""

    def __init__(self) -> None:
        """Initialize the local event detection service."""
        self.ha_url = os.getenv("HOME_ASSISTANT_URL", "http://localhost:8123")
        self.ha_token = os.getenv("HOME_ASSISTANT_TOKEN")
        self.sensitivity = float(os.getenv("EVENT_DETECTION_SENSITIVITY", "0.7"))

        self.active_cameras: Dict[str, bool] = {}
        self.detection_tasks: Dict[str, Any] = {}
        self.event_history: List[DetectionEvent] = []
        self.nest_cameras_service: Optional[Any] = None
        self.llm_service: Optional[Any] = None

        # Event type mapping
        self.event_types = {
            "person": "camera_person",
            "vehicle": "camera_vehicle",
            "animal": "camera_animal",
            "package": "camera_package",
            "motion": "camera_motion",
            "activity": "camera_activity",
        }

    async def initialize(self) -> bool:
        """Initialize event detection service."""
        try:
            # Get references to other services
            from modules.local_models.service import LocalLLMService
            from modules.nest_cameras.service import NestCameraService

            self.nest_cameras_service = NestCameraService()
            self.llm_service = LocalLLMService()

            # Ensure LLM is loaded
            if self.llm_service:
                await self.llm_service.initialize()

            # Get available cameras
            if self.nest_cameras_service:
                cameras = await self.nest_cameras_service.get_available_cameras()
            else:
                cameras = []
            logger.info(
                f"Initialized event detection for {len(cameras)} cameras: {cameras}"
            )

            # Start detection tasks for each camera
            for camera_name in cameras:
                await self.start_camera_detection(camera_name)

            return True

        except Exception as e:
            logger.error(f"Failed to initialize event detection: {e}")
            return False

    async def start_camera_detection(self, camera_name: str) -> None:
        """Start event detection for a specific camera."""
        if camera_name in self.detection_tasks:
            logger.warning(f"Detection already running for {camera_name}")
            return

        logger.info(f"Starting event detection for camera: {camera_name}")
        task = asyncio.create_task(self._detection_loop(camera_name))
        self.detection_tasks[camera_name] = task

    async def stop_camera_detection(self, camera_name: str) -> None:
        """Stop event detection for a specific camera."""
        if camera_name in self.detection_tasks:
            self.detection_tasks[camera_name].cancel()
            del self.detection_tasks[camera_name]
            logger.info(f"Stopped event detection for camera: {camera_name}")

    async def _detection_loop(self, camera_name: str) -> None:
        """Run main detection loop for a camera."""
        logger.info(f"Starting detection loop for {camera_name}")

        previous_frame = None
        last_detection_time: Dict[str, datetime] = {}
        detection_cooldown = 10  # seconds between same event types

        while True:
            try:
                # Get camera frames
                frames: List[Any] = []
                if self.nest_cameras_service is not None:
                    frames = await self.nest_cameras_service.extract_camera_frames(
                        camera_name, num_frames=3
                    )

                if not frames:
                    await asyncio.sleep(5)  # Wait before retrying
                    continue

                current_frame = frames[-1]  # Use latest frame

                # Motion detection using frame difference
                motion_detected = False
                if previous_frame is not None:
                    motion_detected = self._detect_motion(previous_frame, current_frame)

                # If motion detected or periodic check, analyze with Llama-4-Scout
                current_time = datetime.now()
                time_since_last = (
                    current_time
                    - max(last_detection_time.values(), default=datetime.min)
                ).seconds
                should_analyze = (
                    motion_detected or not last_detection_time or time_since_last > 30
                )

                if should_analyze:
                    events = await self._analyze_frame_with_llama(camera_name, frames)

                    for event in events:
                        # Apply cooldown to prevent spam
                        last_time = last_detection_time.get(
                            event.event_type, datetime.min
                        )
                        if (current_time - last_time).seconds > detection_cooldown:
                            await self._send_ha_event(event)
                            last_detection_time[event.event_type] = current_time
                            logger.info(
                                f"Event detected: {event.event_type} on {camera_name}"
                            )

                previous_frame = current_frame.copy()
                await asyncio.sleep(3)  # Check every 3 seconds

            except asyncio.CancelledError:
                logger.info(f"Detection loop cancelled for {camera_name}")
                break
            except Exception as e:
                logger.error(f"Error in detection loop for {camera_name}: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    def _detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """Fast motion detection using frame difference."""
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # Count changed pixels
            changed_pixels: int = int(np.count_nonzero(thresh))
            total_pixels: int = int(thresh.shape[0] * thresh.shape[1])
            change_percentage: float = float(changed_pixels) / float(total_pixels)

            # Motion threshold (adjustable)
            return change_percentage > 0.02  # 2% of pixels changed

        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return False

    async def _analyze_frame_with_llama(
        self, camera_name: str, frames: List[np.ndarray]
    ) -> List[DetectionEvent]:
        """Analyze camera frames using Llama-4-Scout vision."""
        if not self.llm_service:
            return []

        try:
            # Prepare analysis prompt
            analysis_prompt = """
            Analyze these camera images for home security events. Detect and classify:

            1. PERSON - Any human beings visible
            2. VEHICLE - Cars, trucks, motorcycles, bicycles
            3. ANIMAL - Dogs, cats, wildlife
            4. PACKAGE - Deliveries, boxes, packages
            5. ACTIVITY - Notable activities or changes in the scene

            For each detection, provide:
            - Type (person/vehicle/animal/package/activity)
            - Confidence (0.0-1.0)
            - Location description
            - Brief description

            Respond in JSON format:
            {
                "detections": [
                    {
                        "type": "person",
                        "confidence": 0.95,
                        "location": "front porch",
                        "description": "Person standing at front door"
                    }
                ]
            }
            """

            # Analyze with Llama-4-Scout
            result = await self.llm_service.analyze_images(frames, analysis_prompt)

            if result.get("error"):
                logger.error(f"Llama analysis failed: {result['error']}")
                return []

            # Parse response
            content = result.get("content", "")
            events = []

            try:
                # Try to parse JSON response
                if "{" in content and "}" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_str = content[json_start:json_end]
                    analysis_data = json.loads(json_str)

                    for detection in analysis_data.get("detections", []):
                        if detection.get("confidence", 0) >= self.sensitivity:
                            event_type = self.event_types.get(
                                detection.get("type"), "camera_activity"
                            )

                            event = DetectionEvent(
                                event_type=event_type,
                                camera_name=camera_name,
                                confidence=detection.get("confidence", 0.0),
                                timestamp=datetime.now(),
                                zones=[detection.get("location", "unknown")],
                                description=detection.get(
                                    "description", "Event detected"
                                ),
                            )
                            events.append(event)

            except json.JSONDecodeError:
                # Fallback: parse text response
                content_lower = content.lower()
                if any(word in content_lower for word in ["person", "human", "people"]):
                    events.append(
                        DetectionEvent(
                            event_type="camera_person",
                            camera_name=camera_name,
                            confidence=0.8,
                            timestamp=datetime.now(),
                            zones=["detected"],
                            description="Person detected in camera view",
                        )
                    )

            return events

        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return []

    async def _send_ha_event(self, event: DetectionEvent) -> None:
        """Send event to Home Assistant."""
        if not self.ha_token:
            logger.warning("No Home Assistant token configured")
            return

        try:
            # Prepare HA event data
            event_data = {
                "event_type": "ciris_camera_event",
                "event_data": {
                    "device_id": f"camera_{event.camera_name}",
                    "type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "confidence": event.confidence,
                    "zones": event.zones,
                    "description": event.description,
                    "source": "ciris_local_detection",
                },
            }

            # Send to Home Assistant events API
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ha_url}/api/events/ciris_camera_event",
                    json=event_data["event_data"],
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        logger.info(f"Sent {event.event_type} event to Home Assistant")
                        self.event_history.append(event)

                        # Keep only last 100 events
                        if len(self.event_history) > 100:
                            self.event_history = self.event_history[-100:]
                    else:
                        logger.error(f"Failed to send HA event: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Home Assistant event: {e}")

    async def get_event_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent event history."""
        recent_events = self.event_history[-limit:] if limit else self.event_history

        return [
            {
                "event_type": event.event_type,
                "camera_name": event.camera_name,
                "confidence": event.confidence,
                "timestamp": event.timestamp.isoformat(),
                "zones": event.zones,
                "description": event.description,
            }
            for event in recent_events
        ]

    async def get_detection_status(self) -> Dict[str, Any]:
        """Get status of event detection system."""
        return {
            "active_cameras": len(self.detection_tasks),
            "camera_names": list(self.detection_tasks.keys()),
            "total_events": len(self.event_history),
            "sensitivity": self.sensitivity,
            "llm_available": self.llm_service is not None,
            "ha_integration": self.ha_token is not None,
        }
