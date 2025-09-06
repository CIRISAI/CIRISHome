#!/usr/bin/env python3
"""
Jetson Nano Simulator for CIRISHome Development
Simulates the Ollama/LLM API for testing without real hardware
"""

import json
import random
import time

from flask import Flask, jsonify, request

app = Flask(__name__)

# Mock model responses
MOCK_MODELS = {
    "llama-4-scout-int4": {
        "name": "llama-4-scout-int4",
        "modified_at": "2025-01-01T00:00:00Z",
        "size": 4294967296,  # ~4GB
        "digest": "sha256:mock-digest-for-testing",
    }
}

MOCK_RESPONSES = {
    "temperature": "The living room temperature is currently 72.5°F.",
    "lights": "I've turned on the living room lights for you.",
    "weather": "It's currently sunny and 75°F outside.",
    "default": "I'm a simulated Jetson Nano LLM for CIRISHome testing.",
}


@app.route("/api/tags", methods=["GET"])
def list_models():
    """List available models"""
    return jsonify({"models": list(MOCK_MODELS.values())})


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate LLM response"""
    data = request.json
    prompt = data.get("prompt", "").lower()

    # Simulate processing time
    time.sleep(random.uniform(0.5, 2.0))

    # Choose response based on prompt
    if "temperature" in prompt:
        response = MOCK_RESPONSES["temperature"]
    elif "lights" in prompt or "light" in prompt:
        response = MOCK_RESPONSES["lights"]
    elif "weather" in prompt:
        response = MOCK_RESPONSES["weather"]
    else:
        response = MOCK_RESPONSES["default"]

    return jsonify(
        {
            "model": data.get("model", "llama-4-scout-int4"),
            "created_at": "2025-01-01T00:00:00Z",
            "response": response,
            "done": True,
            "context": [1, 2, 3],  # Mock context tokens
            "total_duration": random.randint(1000000000, 3000000000),
            "load_duration": random.randint(100000000, 500000000),
            "prompt_eval_count": len(prompt.split()),
            "prompt_eval_duration": random.randint(200000000, 800000000),
            "eval_count": len(response.split()),
            "eval_duration": random.randint(1500000000, 2500000000),
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat endpoint (alternative API)"""
    data = request.json
    messages = data.get("messages", [])

    if messages:
        last_message = messages[-1].get("content", "").lower()
    else:
        last_message = ""

    # Choose response
    if "temperature" in last_message:
        response = MOCK_RESPONSES["temperature"]
    elif "lights" in last_message or "light" in last_message:
        response = MOCK_RESPONSES["lights"]
    elif "weather" in last_message:
        response = MOCK_RESPONSES["weather"]
    else:
        response = MOCK_RESPONSES["default"]

    return jsonify(
        {
            "model": data.get("model", "llama-4-scout-int4"),
            "created_at": "2025-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": response},
            "done": True,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "gpu_memory": "4.2GB/8.0GB",
            "models_loaded": ["llama-4-scout-int4", "whisper-large-v3", "coqui-tts"],
            "jetson_info": {
                "model": "Jetson Orin Nano Developer Kit",
                "cuda_version": "12.2",
                "jetpack": "6.0",
            },
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Jetson Nano Simulator...")
    print("📡 API available at: http://0.0.0.0:11434")
    print("🧪 Mock models: llama-4-scout-int4")
    app.run(host="0.0.0.0", port=11434, debug=True)
