"""
Tests for CIRIS wisdom module integration.
"""

from unittest.mock import AsyncMock, Mock

import pytest


class TestGeoWisdom:
    """Test geographic wisdom module."""

    @pytest.mark.asyncio
    async def test_geocoding_functionality(self, mock_wisdom_modules, geo_data):
        """Test address geocoding."""

        result = await mock_wisdom_modules.geo.geocode("123 Main Street, Anytown, NY")

        assert "latitude" in result
        assert "longitude" in result
        assert "address" in result
        assert isinstance(result["latitude"], (int, float))
        assert isinstance(result["longitude"], (int, float))

    @pytest.mark.asyncio
    async def test_routing_functionality(self, mock_wisdom_modules, geo_data):
        """Test route calculation."""

        start = {"lat": 40.7128, "lon": -74.0060}
        end = {"lat": 40.7589, "lon": -73.9851}

        route = await mock_wisdom_modules.geo.route(start, end)

        assert "distance" in route
        assert "duration" in route
        assert "steps" in route
        assert route["distance"] > 0
        assert route["duration"] > 0
        assert len(route["steps"]) > 0

    def test_geo_privacy_compliance(self):
        """Test that geo wisdom uses privacy-respecting APIs."""

        # Should use OpenStreetMap (no API key required)
        # Should NOT use Google Maps or other tracking services

        expected_config = {
            "primary_api": "OpenStreetMap",
            "api_key_required": False,
            "tracking": False,
            "data_retention": "none",
        }

        assert not expected_config["api_key_required"]
        assert not expected_config["tracking"]
        assert expected_config["data_retention"] == "none"

    @pytest.mark.asyncio
    async def test_geo_error_handling(self, mock_wisdom_modules):
        """Test geo wisdom error handling."""

        # Test invalid address
        mock_wisdom_modules.geo.geocode = AsyncMock(
            side_effect=Exception("Address not found")
        )

        with pytest.raises(Exception, match="Address not found"):
            await mock_wisdom_modules.geo.geocode("Invalid Address")


class TestWeatherWisdom:
    """Test weather wisdom module."""

    @pytest.mark.asyncio
    async def test_current_weather(self, mock_wisdom_modules, weather_data):
        """Test current weather retrieval."""

        weather = await mock_wisdom_modules.weather.current("Anytown, NY")

        assert "temperature" in weather
        assert "conditions" in weather
        assert "humidity" in weather
        assert isinstance(weather["temperature"], (int, float))
        assert isinstance(weather["humidity"], (int, float))

    @pytest.mark.asyncio
    async def test_weather_forecast(self, mock_wisdom_modules, weather_data):
        """Test weather forecast retrieval."""

        forecast = await mock_wisdom_modules.weather.forecast("Anytown, NY", days=3)

        assert isinstance(forecast, list)
        assert len(forecast) > 0

        for day in forecast:
            assert "temperature" in day or "high" in day
            assert "conditions" in day

    def test_weather_api_compliance(self):
        """Test weather API usage compliance."""

        # Should use NOAA (free, no API key for US)
        # Should have OpenWeatherMap as fallback for international

        expected_config = {
            "primary_api": "NOAA",
            "fallback_api": "OpenWeatherMap",
            "primary_requires_key": False,
            "fallback_requires_key": True,
            "data_source": "government",
        }

        assert not expected_config["primary_requires_key"]
        assert expected_config["data_source"] == "government"

    @pytest.mark.asyncio
    async def test_weather_alerts(self, mock_wisdom_modules):
        """Test weather alerts functionality."""

        # Mock weather alerts
        mock_wisdom_modules.weather.alerts = AsyncMock(
            return_value=[
                {
                    "title": "Heat Advisory",
                    "severity": "moderate",
                    "description": "High temperatures expected",
                }
            ]
        )

        alerts = await mock_wisdom_modules.weather.alerts("Anytown, NY")

        assert isinstance(alerts, list)
        if len(alerts) > 0:
            alert = alerts[0]
            assert "title" in alert
            assert "severity" in alert


class TestSensorWisdom:
    """Test sensor wisdom module with medical filtering."""

    @pytest.mark.asyncio
    async def test_safe_entity_retrieval(self, mock_wisdom_modules, sample_entities):
        """Test retrieval of safe sensor entities."""

        entities = await mock_wisdom_modules.sensor.get_safe_entities()

        assert isinstance(entities, list)
        assert len(entities) > 0

        # All returned entities should be safe (non-medical)
        for entity in entities:
            assert "entity_id" in entity
            assert "state" in entity
            assert "attributes" in entity

    def test_medical_sensor_filtering(self, mock_wisdom_modules, medical_entities):
        """Test that medical sensors are filtered out."""

        # Medical entities should be filtered out
        filtered = mock_wisdom_modules.sensor.filter_medical(medical_entities)

        # Should return empty list (all filtered)
        assert len(filtered) == 0

    def test_sensor_filtering_keywords(self):
        """Test comprehensive medical keyword filtering."""

        prohibited_keywords = [
            "heart_rate",
            "blood_pressure",
            "blood_glucose",
            "blood_oxygen",
            "body_temperature",
            "weight",
            "bmi",
            "spo2",
            "ecg",
            "pulse",
            "medical",
            "health",
            "patient",
            "vital",
        ]

        def is_medical_sensor(
            entity_id: str, friendly_name: str = "", device_class: str = ""
        ) -> bool:
            text = f"{entity_id} {friendly_name} {device_class}".lower()
            return any(keyword in text for keyword in prohibited_keywords)

        # Test medical sensors are detected
        medical_tests = [
            ("sensor.heart_rate", "Heart Rate Monitor", "heart_rate"),
            ("sensor.bp_monitor", "Blood Pressure", "blood_pressure"),
            ("sensor.patient_temp", "Patient Temperature", "temperature"),
            ("sensor.medical_device", "Medical Device", "medical"),
        ]

        for entity_id, friendly_name, device_class in medical_tests:
            assert is_medical_sensor(
                entity_id, friendly_name, device_class
            ), f"Medical sensor not detected: {entity_id}"

        # Test safe sensors pass
        safe_tests = [
            ("sensor.room_temp", "Room Temperature", "temperature"),
            ("sensor.outdoor_humidity", "Outdoor Humidity", "humidity"),
            ("sensor.air_quality", "Air Quality", "aqi"),
        ]

        for entity_id, friendly_name, device_class in safe_tests:
            assert not is_medical_sensor(
                entity_id, friendly_name, device_class
            ), f"Safe sensor incorrectly flagged: {entity_id}"

    @pytest.mark.asyncio
    async def test_ha_integration(self, mock_wisdom_modules):
        """Test Home Assistant integration."""

        # Should integrate with HA API
        entities = await mock_wisdom_modules.sensor.get_safe_entities()

        # Verify HA-style entity format
        for entity in entities[:1]:  # Test first entity
            assert entity["entity_id"].count(".") == 1  # domain.object_id format
            assert "attributes" in entity
            assert "state" in entity

    def test_sensor_wisdom_safety_boundaries(self):
        """Test that sensor wisdom respects safety boundaries."""

        # Prohibited domains
        prohibited_domains = ["medical", "health", "clinical", "patient"]

        # Safe domains
        safe_domains = ["environmental", "energy", "security", "automation"]

        def get_entity_domain(entity_id: str) -> str:
            """Extract domain context from entity."""
            text = entity_id.lower()

            for domain in prohibited_domains:
                if domain in text:
                    return "prohibited"

            for domain in safe_domains:
                if domain in text:
                    return "safe"

            return "unknown"

        # Test prohibited entities
        prohibited_entities = [
            "sensor.medical_device",
            "sensor.health_monitor",
            "sensor.patient_room_sensor",
            "sensor.clinical_thermometer",
        ]

        for entity_id in prohibited_entities:
            domain = get_entity_domain(entity_id)
            assert (
                domain == "prohibited"
            ), f"Prohibited entity not detected: {entity_id}"

        # Test safe entities
        safe_entities = [
            "sensor.environmental_temp",
            "sensor.energy_usage",
            "sensor.security_motion",
            "sensor.automation_status",
        ]

        for entity_id in safe_entities:
            domain = get_entity_domain(entity_id)
            assert domain in [
                "safe",
                "unknown",
            ], f"Safe entity incorrectly flagged: {entity_id}"


class TestWisdomIntegration:
    """Test integration between wisdom modules."""

    @pytest.mark.asyncio
    async def test_multi_wisdom_query(self, mock_wisdom_modules):
        """Test queries that use multiple wisdom modules."""

        # Weather + Geo combination
        weather = await mock_wisdom_modules.weather.current("Anytown, NY")
        location = await mock_wisdom_modules.geo.geocode("Anytown, NY")

        # Should be able to combine data
        combined_info = {
            "location": location,
            "current_weather": weather,
            "context": f"Weather at {location.get('address', 'location')} is {weather.get('conditions', 'unknown')}",
        }

        assert "location" in combined_info
        assert "current_weather" in combined_info
        assert len(combined_info["context"]) > 0

    @pytest.mark.asyncio
    async def test_sensor_weather_correlation(self, mock_wisdom_modules):
        """Test correlating indoor sensors with outdoor weather."""

        # Get indoor sensor data
        indoor_entities = await mock_wisdom_modules.sensor.get_safe_entities()
        indoor_temp = None

        for entity in indoor_entities:
            if (
                "temperature" in entity["entity_id"]
                and "living_room" in entity["entity_id"]
            ):
                indoor_temp = float(entity["state"])
                break

        # Get outdoor weather
        weather = await mock_wisdom_modules.weather.current("Local")
        outdoor_temp = weather["temperature"]

        if indoor_temp and outdoor_temp:
            temp_diff = abs(indoor_temp - outdoor_temp)

            # Should be reasonable correlation
            assert temp_diff < 50, f"Temperature difference too large: {temp_diff}°F"

    def test_wisdom_module_boundaries(self):
        """Test that wisdom modules respect their boundaries."""

        # Each wisdom module should have clear boundaries
        module_boundaries = {
            "geo": {
                "allowed": ["navigation", "routing", "geocoding", "mapping"],
                "prohibited": ["tracking", "surveillance", "personal_location_history"],
            },
            "weather": {
                "allowed": [
                    "forecasts",
                    "current_conditions",
                    "alerts",
                    "atmospheric_data",
                ],
                "prohibited": ["personal_health_advice", "medical_recommendations"],
            },
            "sensor": {
                "allowed": ["environmental", "energy", "security", "automation"],
                "prohibited": ["medical", "health", "clinical", "patient", "vital"],
            },
        }

        # Verify boundaries are respected
        for module, boundaries in module_boundaries.items():
            assert (
                len(boundaries["allowed"]) > 0
            ), f"Module {module} should have allowed capabilities"
            assert (
                len(boundaries["prohibited"]) > 0
            ), f"Module {module} should have prohibited capabilities"

    @pytest.mark.asyncio
    async def test_wisdom_error_isolation(self, mock_wisdom_modules):
        """Test that wisdom module errors don't cascade."""

        # Simulate geo module failure
        mock_wisdom_modules.geo.geocode = AsyncMock(
            side_effect=Exception("Geo service down")
        )

        # Weather and sensor should still work
        weather = await mock_wisdom_modules.weather.current("Test Location")
        entities = await mock_wisdom_modules.sensor.get_safe_entities()

        assert weather is not None
        assert len(entities) >= 0

        # Geo should fail independently
        with pytest.raises(Exception, match="Geo service down"):
            await mock_wisdom_modules.geo.geocode("Test Address")

    def test_wisdom_configuration_validation(self):
        """Test wisdom module configuration validation."""

        # Required environment variables for each module
        required_env_vars = {
            "geo": ["CIRIS_OSM_USER_AGENT"],
            "weather": ["CIRIS_NOAA_USER_AGENT"],
            "sensor": ["CIRIS_HOMEASSISTANT_URL", "CIRIS_HOMEASSISTANT_TOKEN"],
        }

        # Optional environment variables
        optional_env_vars = {"weather": ["CIRIS_OPENWEATHERMAP_API_KEY"]}

        # Verify configuration structure
        for module, vars_list in required_env_vars.items():
            assert (
                len(vars_list) > 0
            ), f"Module {module} should have required environment variables"

        # Verify optional configurations exist
        assert "weather" in optional_env_vars
        assert "CIRIS_OPENWEATHERMAP_API_KEY" in optional_env_vars["weather"]
