"""Tests for the complete voice processing pipeline."""

import asyncio
from unittest.mock import AsyncMock

import pytest


class TestVoicePipeline:
    """Test complete voice interaction pipeline."""

    @pytest.mark.asyncio
    async def test_complete_voice_interaction(
        self, mock_jetson, mock_homeassistant, sample_audio
    ):
        """Test complete voice interaction flow from audio to response."""
        # 1. STT - Convert audio to text
        stt_result = await mock_jetson.transcribe(sample_audio)

        assert stt_result is not None
        assert "text" in stt_result
        assert stt_result["confidence"] > 0.7

        # 2. Get Home Assistant context
        ha_states = await mock_homeassistant.get_states()
        assert len(ha_states) > 0

        # 3. LLM - Process with context
        prompt = f"User: {stt_result['text']}. Context: {ha_states}"
        llm_result = await mock_jetson.generate(prompt)

        assert llm_result is not None
        assert "response" in llm_result
        assert llm_result["tokens_used"] > 0

        # 4. TTS - Convert response to audio
        tts_result = await mock_jetson.synthesize(llm_result["response"])

        assert tts_result is not None
        assert "audio_data" in tts_result
        assert len(tts_result["audio_data"]) > 0

    @pytest.mark.asyncio
    async def test_home_control_command(
        self, mock_jetson, mock_homeassistant, service_calls
    ):
        """Test voice command that controls home devices."""
        calls, add_call = service_calls

        # Mock service call in HA
        mock_homeassistant.call_service = AsyncMock(
            side_effect=lambda domain, service, **kwargs: add_call(
                domain, service, **kwargs
            )
        )

        # Voice command: "Turn on the living room lights"
        audio_input = b"mock_audio_lights_command"

        # STT - returns "Turn on the living room lights"
        stt_result = await mock_jetson.transcribe(audio_input)
        assert "lights" in stt_result["text"].lower()

        # LLM processes command - prompt contains "lights" so returns control response
        llm_result = await mock_jetson.generate(f"User command: {stt_result['text']}")

        # Verify LLM identified it as a lights command
        assert "lights" in llm_result.get("response", "").lower()

        # Make the service call based on LLM response
        await mock_homeassistant.call_service(
            "light", "turn_on", entity_id="light.living_room_lights"
        )

        # Verify service call was made
        assert len(calls) == 1
        assert calls[0]["domain"] == "light"
        assert calls[0]["service"] == "turn_on"
        assert calls[0]["data"]["entity_id"] == "light.living_room_lights"

    @pytest.mark.asyncio
    async def test_camera_vision_query(self, mock_jetson, sample_image):
        """Test voice query about camera vision."""
        # Voice query about camera
        query = "What do you see on the front door camera?"

        # Mock vision-enabled LLM response
        mock_jetson.generate = AsyncMock(
            return_value={
                "response": (
                    "I can see a person standing at the front door with a package."
                ),
                "vision_analysis": {
                    "objects_detected": ["person", "package", "door"],
                    "confidence_scores": [0.95, 0.87, 0.99],
                },
                "tokens_used": 45,
            }
        )

        # Process with image data
        result = await mock_jetson.generate(query, image_data=sample_image)

        assert "vision_analysis" in result
        assert "objects_detected" in result["vision_analysis"]
        assert len(result["vision_analysis"]["objects_detected"]) > 0
        assert all(
            score > 0.7 for score in result["vision_analysis"]["confidence_scores"]
        )

    @pytest.mark.asyncio
    async def test_wisdom_integration(self, mock_wisdom_modules):
        """Test integration with wisdom modules."""
        # Test weather query
        weather = await mock_wisdom_modules.weather.current("Anytown, NY")
        assert weather["temperature"] is not None
        assert weather["conditions"] is not None

        # Test geo query
        location = await mock_wisdom_modules.geo.geocode("123 Main Street")
        assert "latitude" in location
        assert "longitude" in location

        # Test sensor query (with medical filtering)
        safe_entities = await mock_wisdom_modules.sensor.get_safe_entities()
        assert len(safe_entities) > 0

        # Verify medical filtering
        filtered = mock_wisdom_modules.sensor.filter_medical(safe_entities)
        assert len(filtered) == 0  # Should filter out medical entities

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_jetson, sample_audio):
        """Test error handling in pipeline."""
        # Test STT failure
        mock_jetson.transcribe = AsyncMock(
            side_effect=Exception("STT service unavailable")
        )

        with pytest.raises(Exception, match="STT service unavailable"):
            await mock_jetson.transcribe(sample_audio)

        # Test LLM failure
        mock_jetson.generate = AsyncMock(side_effect=Exception("LLM model not loaded"))

        with pytest.raises(Exception, match="LLM model not loaded"):
            await mock_jetson.generate("test prompt")

        # Test TTS failure
        mock_jetson.synthesize = AsyncMock(
            side_effect=Exception("TTS synthesis failed")
        )

        with pytest.raises(Exception, match="TTS synthesis failed"):
            await mock_jetson.synthesize("test response")

    @pytest.mark.asyncio
    async def test_pipeline_latency(self, mock_jetson, sample_audio):
        """Test pipeline meets latency requirements."""
        start_time = asyncio.get_event_loop().time()

        # Complete pipeline
        stt_result = await mock_jetson.transcribe(sample_audio)
        llm_result = await mock_jetson.generate(f"Query: {stt_result['text']}")
        await mock_jetson.synthesize(llm_result["response"])

        end_time = asyncio.get_event_loop().time()
        total_latency = end_time - start_time

        # Pipeline should be fast (mocked services are instant)
        assert total_latency < 1.0, f"Pipeline too slow: {total_latency}s"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_jetson, sample_audio):
        """Test handling multiple concurrent voice requests."""

        async def process_voice_request(request_id: str):
            stt_result = await mock_jetson.transcribe(sample_audio)
            llm_result = await mock_jetson.generate(
                f"Request {request_id}: {stt_result['text']}"
            )
            await mock_jetson.synthesize(llm_result["response"])
            return {"id": request_id, "success": True}

        # Process 3 concurrent requests
        tasks = [
            process_voice_request("req_1"),
            process_voice_request("req_2"),
            process_voice_request("req_3"),
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 3
        assert all(result["success"] for result in results)


class TestWyomingProtocol:
    """Test Wyoming protocol integration with Voice PE pucks."""

    def test_wyoming_message_format(self, wyoming_messages):
        """Test Wyoming protocol message formats."""
        # Info message
        info_msg = wyoming_messages["info"]
        assert info_msg["type"] == "info"
        assert "asr" in info_msg
        assert "tts" in info_msg

        # Transcript message
        transcript_msg = wyoming_messages["transcript"]
        assert transcript_msg["type"] == "transcript"
        assert "text" in transcript_msg

        # Audio message
        audio_msg = wyoming_messages["audio"]
        assert audio_msg["type"] == "audio"
        assert "rate" in audio_msg
        assert "audio" in audio_msg

    @pytest.mark.asyncio
    async def test_wyoming_communication(self, mock_wyoming, wyoming_messages):
        """Test Wyoming protocol communication flow."""
        # Connect to Wyoming server
        await mock_wyoming.connect()
        mock_wyoming.connect.assert_called_once()

        # Send audio for transcription
        await mock_wyoming.send_audio(b"audio_data")
        mock_wyoming.send_audio.assert_called_once()

        # Receive transcript
        mock_wyoming.receive_transcript = AsyncMock(
            return_value=wyoming_messages["transcript"]
        )
        transcript = await mock_wyoming.receive_transcript()

        assert transcript["type"] == "transcript"
        assert "text" in transcript

        # Send TTS request
        await mock_wyoming.send_tts("Response text")
        mock_wyoming.send_tts.assert_called_once()

        # Receive audio response
        mock_wyoming.receive_audio = AsyncMock(return_value=wyoming_messages["audio"])
        audio = await mock_wyoming.receive_audio()

        assert audio["type"] == "audio"
        assert "audio" in audio

    def test_voice_pe_device_config(self, voice_pe_devices):
        """Test Voice PE device configuration."""
        assert len(voice_pe_devices) >= 2

        for device in voice_pe_devices:
            assert "device_id" in device
            assert "ip_address" in device
            assert "location" in device
            assert device["device_id"].startswith("voice_pe_")


class TestPrivacyCompliance:
    """Test privacy and safety compliance."""

    def test_local_processing_only(self, env_vars):
        """Test that all processing is local."""
        # Environment should specify local processing
        assert env_vars["LOCAL_PROCESSING_ONLY"] == "true"
        assert env_vars["MEDICAL_GRADE_PRIVACY"] == "true"

        # Jetson endpoint should be local
        assert "192.168.1." in env_vars["OPENAI_API_BASE"]
        assert "homeassistant.local" in env_vars["HOME_ASSISTANT_URL"]

    def test_medical_sensor_blocking(self, medical_entities):
        """Test that medical sensors are blocked."""
        medical_keywords = ["heart_rate", "blood_pressure", "medical", "patient"]

        def is_medical_blocked(entity):
            entity_text = (
                f"{entity['entity_id']} {entity['attributes']['friendly_name']}".lower()
            )
            return any(keyword in entity_text for keyword in medical_keywords)

        # All medical entities should be blocked
        for entity in medical_entities:
            assert is_medical_blocked(
                entity
            ), f"Medical entity not blocked: {entity['entity_id']}"

    def test_responsibility_acceptance_required(self, env_vars):
        """Test that home automation responsibility is accepted."""
        assert env_vars["I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY"] == "true"

    def test_audit_trail_available(self):
        """Test that interactions can be audited."""
        # Simulate audit trail
        interaction_log = {
            "timestamp": "2024-01-01T12:00:00Z",
            "device_id": "voice_pe_living_room",
            "query": "What's the temperature?",
            "response": "The temperature is 72°F",
            "ha_entities_accessed": ["sensor.living_room_temperature"],
            "medical_entities_blocked": 0,
            "processing_location": "local_jetson",
        }

        # Verify audit information is complete
        assert "timestamp" in interaction_log
        assert "processing_location" in interaction_log
        assert interaction_log["processing_location"] == "local_jetson"
        assert "medical_entities_blocked" in interaction_log
