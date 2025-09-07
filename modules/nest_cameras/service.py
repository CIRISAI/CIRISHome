"""
Nest Camera Service - Access Nest cameras via WebRTC/go2rtc integration.

Integrates with Home Assistant's WebRTC Camera custom component and go2rtc
server to provide multi-modal vision processing for Nest cameras.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NestCameraService:
    """Service for accessing Nest cameras via WebRTC/go2rtc."""

    def __init__(self):
        """Initialize the Nest camera service."""
        self.go2rtc_url = os.getenv("GO2RTC_SERVER_URL", "http://127.0.0.1:8554")
        self.camera_urls = self._parse_camera_urls()
        self.active_streams = {}

    def _parse_camera_urls(self) -> Dict[str, str]:
        """Parse camera URLs from environment variable."""
        urls_env = os.getenv("WEBRTC_CAMERA_URLS", "")
        camera_urls = {}

        # Expected format: "front_door:rtsp://127.0.0.1:8554/front_door,
        # backyard:rtsp://127.0.0.1:8554/backyard"
        if urls_env:
            for camera_def in urls_env.split(","):
                if ":" in camera_def:
                    name, url = camera_def.split(":", 1)
                    camera_urls[name.strip()] = url.strip()

        logger.info(f"Configured {len(camera_urls)} Nest cameras via go2rtc")
        return camera_urls

    async def get_camera_stream(self, camera_name: str) -> Optional[str]:
        """Get RTSP stream URL for a specific camera."""
        if camera_name not in self.camera_urls:
            logger.error(f"Camera {camera_name} not found in configuration")
            return None

        stream_url = self.camera_urls[camera_name]
        logger.info(f"Accessing Nest camera {camera_name} via {stream_url}")
        return stream_url

    async def get_available_cameras(self) -> List[str]:
        """Get list of available camera names."""
        return list(self.camera_urls.keys())

    async def analyze_camera_feed(
        self, camera_name: str, duration_seconds: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze camera feed for motion detection and object recognition.

        Uses OpenCV with Jetson GPU acceleration when available.
        """
        stream_url = await self.get_camera_stream(camera_name)
        if not stream_url:
            return {"error": f"Camera {camera_name} not available"}

        try:
            # Open video stream
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                return {"error": f"Could not open stream for {camera_name}"}

            analysis_results = {
                "camera_name": camera_name,
                "duration_seconds": duration_seconds,
                "frames_analyzed": 0,
                "motion_detected": False,
                "objects_detected": [],
                "average_brightness": 0.0,
            }

            frame_count = 0
            brightness_sum = 0
            previous_frame = None
            motion_threshold = 5000  # Adjust based on camera sensitivity

            # Analyze frames for specified duration
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate brightness
                brightness = np.mean(gray)
                brightness_sum += brightness

                # Motion detection using frame difference
                if previous_frame is not None:
                    diff = cv2.absdiff(previous_frame, gray)
                    non_zero_count = np.count_nonzero(diff > 30)

                    if non_zero_count > motion_threshold:
                        analysis_results["motion_detected"] = True
                        logger.debug(
                            f"Motion detected in {camera_name}: "
                            f"{non_zero_count} pixels changed"
                        )

                previous_frame = gray.copy()

                # Add small delay to prevent overwhelming the stream
                await asyncio.sleep(0.1)

            cap.release()

            analysis_results["frames_analyzed"] = frame_count
            if frame_count > 0:
                analysis_results["average_brightness"] = brightness_sum / frame_count

            logger.info(f"Analyzed {frame_count} frames from {camera_name}")
            return analysis_results

        except Exception as e:
            logger.error(f"Error analyzing camera {camera_name}: {e}")
            return {"error": str(e)}

    async def detect_motion(self, camera_name: str, sensitivity: float = 0.5) -> bool:
        """
        Quick motion detection on camera feed.

        Returns True if motion detected in last few seconds.
        """
        analysis = await self.analyze_camera_feed(camera_name, duration_seconds=3)
        return analysis.get("motion_detected", False)

    async def extract_camera_frames(
        self, camera_name: str, num_frames: int = 5
    ) -> List[np.ndarray]:
        """
        Extract specific number of frames from camera feed.

        Useful for multi-modal analysis with vision pipeline.
        """
        stream_url = await self.get_camera_stream(camera_name)
        if not stream_url:
            return []

        frames = []
        try:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.error(f"Could not open stream for {camera_name}")
                return []

            for i in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    await asyncio.sleep(0.2)  # Small delay between frames
                else:
                    break

            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {camera_name}")

        except Exception as e:
            logger.error(f"Error extracting frames from {camera_name}: {e}")

        return frames

    async def get_camera_status(self) -> Dict[str, Any]:
        """Get status of all configured cameras."""
        status = {
            "go2rtc_server": self.go2rtc_url,
            "total_cameras": len(self.camera_urls),
            "cameras": {},
        }

        for camera_name in self.camera_urls.keys():
            try:
                # Quick connection test
                cap = cv2.VideoCapture(self.camera_urls[camera_name])
                is_online = cap.isOpened()
                cap.release()

                status["cameras"][camera_name] = {
                    "url": self.camera_urls[camera_name],
                    "online": is_online,
                    "type": "nest_via_webrtc",
                }
            except Exception as e:
                status["cameras"][camera_name] = {
                    "url": self.camera_urls[camera_name],
                    "online": False,
                    "error": str(e),
                }

        return status
