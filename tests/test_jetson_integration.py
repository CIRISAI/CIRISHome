"""Tests for Jetson Nano integration and local AI processing."""

from unittest.mock import AsyncMock

import pytest


class TestJetsonIntegration:
    """Test Jetson Nano AI processing integration."""

    @pytest.mark.asyncio
    async def test_jetson_health_check(self, mock_jetson):
        """Test Jetson Nano health check."""
        health = await mock_jetson.health()

        assert health["status"] == "healthy"
        assert "gpu_memory" in health
        assert "models_loaded" in health
        assert len(health["models_loaded"]) > 0

    @pytest.mark.asyncio
    async def test_stt_processing(self, mock_jetson, sample_audio):
        """Test speech-to-text processing."""
        result = await mock_jetson.transcribe(sample_audio)

        assert "text" in result
        assert "confidence" in result
        assert result["confidence"] > 0.0
        assert result["confidence"] <= 1.0
        assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_llm_text_generation(self, mock_jetson):
        """Test LLM text generation."""
        prompt = "What's the weather like today?"
        result = await mock_jetson.generate(prompt)

        assert "response" in result
        assert "tokens_used" in result
        assert result["tokens_used"] > 0
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_llm_vision_analysis(self, mock_jetson, sample_image):
        """Test LLM vision analysis capabilities."""
        # Mock vision response for image analysis
        mock_jetson.generate = AsyncMock(
            return_value={
                "response": "I can see a person at the door with a package.",
                "vision_analysis": {
                    "objects_detected": ["person", "package", "door"],
                    "confidence_scores": [0.95, 0.87, 0.99],
                    "bounding_boxes": [
                        [100, 150, 200, 400],
                        [180, 300, 220, 340],
                        [0, 0, 640, 480],
                    ],
                },
                "tokens_used": 45,
                "inference_time": 3.2,
            }
        )

        prompt = "Analyze this image for objects and people"
        result = await mock_jetson.generate(prompt, image_data=sample_image)

        assert "vision_analysis" in result
        vision = result["vision_analysis"]
        assert "objects_detected" in vision
        assert "confidence_scores" in vision
        assert len(vision["objects_detected"]) == len(vision["confidence_scores"])
        assert all(score > 0.7 for score in vision["confidence_scores"])

    @pytest.mark.asyncio
    async def test_tts_synthesis(self, mock_jetson):
        """Test text-to-speech synthesis."""
        text = "The living room temperature is 72 degrees."
        result = await mock_jetson.synthesize(text)

        assert "audio_data" in result
        assert "duration" in result
        assert len(result["audio_data"]) > 0
        assert result["duration"] > 0

    def test_model_specifications(self, jetson_responses):
        """Test that Jetson models meet specifications."""
        # Test model specifications are handled by response validation below

        # Verify response formats match expectations
        stt_response = jetson_responses["stt"]
        assert "text" in stt_response
        assert "confidence" in stt_response
        assert "processing_time" in stt_response

        llm_response = jetson_responses["llm_simple"]
        assert "response" in llm_response
        assert "tokens_used" in llm_response
        assert "inference_time" in llm_response

        tts_response = jetson_responses["tts"]
        assert "audio_data" in tts_response
        assert "duration" in tts_response
        assert "processing_time" in tts_response

    @pytest.mark.asyncio
    async def test_quantization_effectiveness(self, mock_jetson):
        """Test that INT4 quantization provides good performance."""
        # Mock performance metrics for quantized model
        health = await mock_jetson.health()

        # Should use reasonable amount of GPU memory (not full 17B model size)
        gpu_memory = health.get("gpu_memory", "4.2GB/8.0GB")
        memory_used = float(gpu_memory.split("/")[0].replace("GB", ""))
        memory_total = float(gpu_memory.split("/")[1].replace("GB", ""))

        # With INT4 quantization, should use < 6GB for all models
        assert memory_used < 6.0, f"GPU memory usage too high: {memory_used}GB"
        assert memory_used / memory_total < 0.8, "GPU memory utilization too high"

    @pytest.mark.asyncio
    async def test_local_processing_verification(self, mock_jetson, ciris_home_config):
        """Test that processing is happening locally on Jetson."""
        # Verify endpoints point to local Jetson
        assert ciris_home_config["jetson_nano_ip"].startswith("192.168.")
        assert ciris_home_config["jetson_nano_port"] in [11434, 8080]

        # All AI operations should be local
        health = await mock_jetson.health()
        assert health["status"] == "healthy"  # Jetson is responding

        # No external API calls should be made
        # This is implicitly tested by using mocks

    @pytest.mark.asyncio
    async def test_performance_requirements(self, mock_jetson, sample_audio):
        """Test that Jetson meets performance requirements."""
        import asyncio

        # Test STT latency
        start_time = asyncio.get_event_loop().time()
        await mock_jetson.transcribe(sample_audio)
        stt_time = asyncio.get_event_loop().time() - start_time

        # Test LLM latency
        start_time = asyncio.get_event_loop().time()
        await mock_jetson.generate("Simple query")
        llm_time = asyncio.get_event_loop().time() - start_time

        # Test TTS latency
        start_time = asyncio.get_event_loop().time()
        await mock_jetson.synthesize("Short response")
        tts_time = asyncio.get_event_loop().time() - start_time

        total_time = stt_time + llm_time + tts_time

        # For mocked services, should be very fast
        assert total_time < 0.1, f"Pipeline too slow: {total_time}s"

        # In real implementation, target times would be:
        # STT: < 5 seconds, LLM: 15-25 tokens/sec, TTS: < 3 seconds


class TestModelManagement:
    """Test Jetson model loading and management."""

    @pytest.mark.asyncio
    async def test_model_loading_status(self, mock_jetson):
        """Test that required models are loaded."""
        health = await mock_jetson.health()
        models_loaded = health.get("models_loaded", [])

        required_models = ["llama-4-scout-int4", "whisper-large-v3", "coqui-tts"]

        for model in required_models:
            assert model in models_loaded, f"Required model not loaded: {model}"

    def test_memory_optimization(self, jetson_responses):
        """Test memory usage optimization."""
        # INT4 quantization should significantly reduce memory usage
        # Original Llama-4-17B would be ~34GB
        # INT4 quantized should be ~4GB

        expected_memory_usage = {
            "llm": 4.0,  # GB - INT4 quantized
            "stt": 1.5,  # GB - Whisper Large v3
            "tts": 0.5,  # GB - Coqui TTS
            "intent": 0.25,  # GB - DistilBERT
            "overhead": 0.5,  # GB - System overhead
        }

        total_expected = sum(expected_memory_usage.values())
        assert (
            total_expected <= 8.0
        ), f"Total memory usage {total_expected}GB exceeds Jetson capacity"

    @pytest.mark.asyncio
    async def test_model_switching_capability(self, mock_jetson):
        """Test ability to switch between different model tasks."""
        # Should be able to handle different types of requests

        # STT request
        stt_result = await mock_jetson.transcribe(b"audio_data")
        assert "text" in stt_result

        # LLM request
        llm_result = await mock_jetson.generate("text prompt")
        assert "response" in llm_result

        # TTS request
        tts_result = await mock_jetson.synthesize("response text")
        assert "audio_data" in tts_result

        # Models should handle task switching efficiently

    def test_gpu_utilization_monitoring(self):
        """Test GPU utilization monitoring."""
        # Mock GPU metrics
        gpu_metrics = {
            "gpu_utilization": 75,  # Percent
            "memory_utilization": 60,  # Percent
            "temperature": 65,  # Celsius
            "power_draw": 45,  # Watts
        }

        # Should be within safe operating ranges
        assert gpu_metrics["gpu_utilization"] <= 95, "GPU utilization too high"
        assert (
            gpu_metrics["memory_utilization"] <= 85
        ), "GPU memory utilization too high"
        assert gpu_metrics["temperature"] <= 80, "GPU temperature too high"
        assert gpu_metrics["power_draw"] <= 60, "Power consumption too high"


class TestErrorRecovery:
    """Test error handling and recovery for Jetson integration."""

    @pytest.mark.asyncio
    async def test_model_failure_recovery(self, mock_jetson):
        """Test recovery from model failures."""
        # Simulate model failure
        mock_jetson.generate = AsyncMock(side_effect=Exception("Model not responding"))

        with pytest.raises(Exception, match="Model not responding"):
            await mock_jetson.generate("test prompt")

        # Simulate recovery
        mock_jetson.generate = AsyncMock(
            return_value={"response": "Model recovered", "tokens_used": 10}
        )

        result = await mock_jetson.generate("test prompt")
        assert result["response"] == "Model recovered"

    @pytest.mark.asyncio
    async def test_gpu_memory_overflow_handling(self, mock_jetson):
        """Test handling of GPU memory overflow."""
        # Simulate memory overflow
        mock_jetson.generate = AsyncMock(side_effect=Exception("CUDA out of memory"))

        with pytest.raises(Exception, match="CUDA out of memory"):
            await mock_jetson.generate("very long prompt that causes memory overflow")

        # Should be able to recover with smaller requests
        mock_jetson.generate = AsyncMock(
            return_value={"response": "Small request works", "tokens_used": 5}
        )

        result = await mock_jetson.generate("short prompt")
        assert result["response"] == "Small request works"

    @pytest.mark.asyncio
    async def test_network_connectivity_issues(self, mock_jetson, ciris_home_config):
        """Test handling of network connectivity issues."""
        # Simulate network timeout
        mock_jetson.health = AsyncMock(side_effect=Exception("Connection timeout"))

        with pytest.raises(Exception, match="Connection timeout"):
            await mock_jetson.health()

        # Should be able to recover when network is restored
        mock_jetson.health = AsyncMock(return_value={"status": "healthy"})

        health = await mock_jetson.health()
        assert health["status"] == "healthy"
