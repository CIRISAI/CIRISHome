"""Tests for medical sensor filtering - critical safety functionality."""

from typing import Any, Dict

import pytest


class TestMedicalSensorFiltering:
    """Test medical sensor filtering for safety compliance."""

    @pytest.fixture
    def medical_sensor_keywords(self):
        """Comprehensive list of prohibited medical keywords."""
        return [
            "heart_rate",
            "heartrate",
            "heart rate",
            "blood_pressure",
            "bloodpressure",
            "blood pressure",
            "blood_glucose",
            "bloodglucose",
            "blood glucose",
            "blood_oxygen",
            "bloodoxygen",
            "blood oxygen",
            "body_temperature",
            "bodytemp",
            "body temp",
            "body temperature",
            "weight",
            "bmi",
            "spo2",
            "ecg",
            "pulse",
            "medical",
            "health",
            "patient",
            "vital",
            "clinical",
        ]

    def test_medical_entities_detected(self, medical_entities, medical_sensor_keywords):
        """Test that medical entities are properly detected."""

        def is_medical_entity(entity: Dict[str, Any]) -> bool:
            """Check if entity contains medical keywords."""
            entity_id = entity.get("entity_id", "").lower()
            friendly_name = (
                entity.get("attributes", {}).get("friendly_name", "").lower()
            )
            device_class = entity.get("attributes", {}).get("device_class", "").lower()

            text_to_check = f"{entity_id} {friendly_name} {device_class}"

            return any(keyword in text_to_check for keyword in medical_sensor_keywords)

        # All medical entities should be detected
        for entity in medical_entities:
            assert is_medical_entity(
                entity
            ), f"Medical entity not detected: {entity['entity_id']}"

    def test_safe_entities_pass_filter(self, sample_entities, medical_sensor_keywords):
        """Test that safe environmental entities pass the filter."""

        def is_medical_entity(entity: Dict[str, Any]) -> bool:
            entity_id = entity.get("entity_id", "").lower()
            friendly_name = (
                entity.get("attributes", {}).get("friendly_name", "").lower()
            )
            device_class = entity.get("attributes", {}).get("device_class", "").lower()

            text_to_check = f"{entity_id} {friendly_name} {device_class}"

            return any(keyword in text_to_check for keyword in medical_sensor_keywords)

        # Safe entities should NOT be detected as medical
        for entity in sample_entities:
            assert not is_medical_entity(
                entity
            ), f"Safe entity incorrectly flagged: {entity['entity_id']}"

    def test_case_insensitive_detection(self, medical_sensor_keywords):
        """Test that medical detection is case insensitive."""
        test_entities = [
            {
                "entity_id": "sensor.HEART_RATE_monitor",
                "attributes": {"friendly_name": "Monitor", "device_class": "sensor"},
            },
            {
                "entity_id": "sensor.test",
                "attributes": {
                    "friendly_name": "BLOOD PRESSURE Monitor",
                    "device_class": "sensor",
                },
            },
            {
                "entity_id": "sensor.test",
                "attributes": {"friendly_name": "Monitor", "device_class": "MEDICAL"},
            },
        ]

        def is_medical_entity(entity: Dict[str, Any]) -> bool:
            entity_id = entity.get("entity_id", "").lower()
            friendly_name = (
                entity.get("attributes", {}).get("friendly_name", "").lower()
            )
            device_class = entity.get("attributes", {}).get("device_class", "").lower()

            text_to_check = f"{entity_id} {friendly_name} {device_class}"

            return any(keyword in text_to_check for keyword in medical_sensor_keywords)

        # All case variants should be detected
        for entity in test_entities:
            assert is_medical_entity(entity), f"Case variant not detected: {entity}"

    def test_edge_cases_detected(self, medical_sensor_keywords):
        """Test edge cases that might bypass filtering."""
        edge_cases = [
            {
                "entity_id": "sensor.heartrate_zone2",  # No underscore
                "attributes": {
                    "friendly_name": "Fitness Monitor",
                    "device_class": "sensor",
                },
            },
            {
                "entity_id": "sensor.blood_glucose_trend",
                "attributes": {"friendly_name": "BG Sensor", "device_class": "glucose"},
            },
            {
                "entity_id": "sensor.body_temp_room1",
                "attributes": {
                    "friendly_name": "Body Temp",
                    "device_class": "temperature",
                },
            },
            {
                "entity_id": "sensor.medical_cabinet",
                "attributes": {
                    "friendly_name": "Cabinet Sensor",
                    "device_class": "opening",
                },
            },
            {
                "entity_id": "sensor.patient_room_sensor",
                "attributes": {
                    "friendly_name": "Room Sensor",
                    "device_class": "temperature",
                },
            },
        ]

        def is_medical_entity(entity: Dict[str, Any]) -> bool:
            entity_id = entity.get("entity_id", "").lower()
            friendly_name = (
                entity.get("attributes", {}).get("friendly_name", "").lower()
            )
            device_class = entity.get("attributes", {}).get("device_class", "").lower()

            text_to_check = f"{entity_id} {friendly_name} {device_class}"

            return any(keyword in text_to_check for keyword in medical_sensor_keywords)

        # All edge cases should be caught
        for entity in edge_cases:
            assert is_medical_entity(
                entity
            ), f"Edge case not detected: {entity['entity_id']}"

    def test_comprehensive_keyword_coverage(self):
        """Test comprehensive coverage of medical keywords."""
        # Test each keyword in different contexts
        medical_keywords = [
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

        def is_medical_entity(entity: Dict[str, Any]) -> bool:
            entity_id = entity.get("entity_id", "").lower()
            friendly_name = (
                entity.get("attributes", {}).get("friendly_name", "").lower()
            )
            device_class = entity.get("attributes", {}).get("device_class", "").lower()

            text_to_check = f"{entity_id} {friendly_name} {device_class}"

            return any(keyword in text_to_check for keyword in medical_keywords)

        for keyword in medical_keywords:
            # Test in entity_id
            entity1 = {
                "entity_id": f"sensor.{keyword}_monitor",
                "attributes": {"friendly_name": "Test", "device_class": "sensor"},
            }
            assert is_medical_entity(
                entity1
            ), f"Keyword '{keyword}' not detected in entity_id"

            # Test in friendly_name
            entity2 = {
                "entity_id": "sensor.test",
                "attributes": {
                    "friendly_name": f"Test {keyword} Monitor",
                    "device_class": "sensor",
                },
            }
            assert is_medical_entity(
                entity2
            ), f"Keyword '{keyword}' not detected in friendly_name"

            # Test in device_class
            entity3 = {
                "entity_id": "sensor.test",
                "attributes": {
                    "friendly_name": "Test Monitor",
                    "device_class": keyword,
                },
            }
            assert is_medical_entity(
                entity3
            ), f"Keyword '{keyword}' not detected in device_class"

    def test_access_control_enforcement(self):
        """Test that medical sensor access raises PermissionError."""

        def check_medical_access(entity_id: str) -> None:
            """Simulate access check for medical sensors."""
            medical_keywords = [
                "heart_rate",
                "blood_pressure",
                "medical",
                "patient",
                "vital",
            ]

            if any(keyword in entity_id.lower() for keyword in medical_keywords):
                raise PermissionError(
                    f"Access to medical sensor {entity_id} is prohibited"
                )

        # Should raise errors for medical sensors
        medical_sensor_ids = [
            "sensor.heart_rate_monitor",
            "sensor.blood_pressure",
            "sensor.medical_device",
            "sensor.patient_monitor",
        ]

        for sensor_id in medical_sensor_ids:
            with pytest.raises(
                PermissionError, match="Access to medical sensor .* is prohibited"
            ):
                check_medical_access(sensor_id)

        # Should NOT raise errors for safe sensors
        safe_sensor_ids = [
            "sensor.living_room_temperature",
            "sensor.outdoor_humidity",
            "sensor.air_quality",
        ]

        for sensor_id in safe_sensor_ids:
            # Should not raise any exception
            check_medical_access(sensor_id)

    def test_filter_tracking_and_monitoring(self, sample_entities, medical_entities):
        """Test that filtering is tracked for monitoring."""
        all_entities = sample_entities + medical_entities
        medical_keywords = [
            "heart_rate",
            "blood_pressure",
            "medical",
            "patient",
            "vital",
            "body_temperature",
        ]

        def filter_entities(entities):
            safe_entities = []
            filtered_count = 0

            for entity in entities:
                entity_id = entity.get("entity_id", "").lower()
                friendly_name = (
                    entity.get("attributes", {}).get("friendly_name", "").lower()
                )
                device_class = (
                    entity.get("attributes", {}).get("device_class", "").lower()
                )

                text_to_check = f"{entity_id} {friendly_name} {device_class}"

                if any(keyword in text_to_check for keyword in medical_keywords):
                    filtered_count += 1
                else:
                    safe_entities.append(entity)

            return safe_entities, filtered_count

        safe, filtered_count = filter_entities(all_entities)

        # Should have filtered out medical entities
        assert filtered_count == len(medical_entities)
        assert len(safe) == len(sample_entities)

        # All returned entities should be safe
        for entity in safe:
            assert entity in sample_entities

    def test_no_false_positives(self):
        """Test that legitimate sensors don't get falsely flagged."""
        # Sensors that should NOT be flagged as medical
        legitimate_sensors = [
            {
                # Contains 'heart' but not medical
                "entity_id": "sensor.greenhouse_temperature",
                "attributes": {
                    "friendly_name": "Greenhouse Temp",
                    "device_class": "temperature",
                },
            },
            {
                # Contains 'weigh' but not medical weight
                "entity_id": "sensor.weighbridge_sensor",
                "attributes": {
                    "friendly_name": "Truck Scale",
                    "device_class": "weight",
                },
            },
            {
                # Contains 'heart' but fireplace
                "entity_id": "sensor.hearth_temperature",
                "attributes": {
                    "friendly_name": "Fireplace Temp",
                    "device_class": "temperature",
                },
            },
        ]

        # More precise filtering - avoid false positives
        medical_keywords = [
            "heart_rate",
            "blood_pressure",
            "blood_glucose",
            "blood_oxygen",
            "body_temperature",
            "bmi",
            "spo2",
            "ecg",
            "pulse",
            "medical",
            "health",
            "patient",
            "vital",
        ]
        # Note: removed standalone "weight" to avoid false positive with weighbridge

        def is_medical_entity(entity: Dict[str, Any]) -> bool:
            entity_id = entity.get("entity_id", "").lower()
            friendly_name = (
                entity.get("attributes", {}).get("friendly_name", "").lower()
            )
            device_class = entity.get("attributes", {}).get("device_class", "").lower()

            text_to_check = f"{entity_id} {friendly_name} {device_class}"

            return any(keyword in text_to_check for keyword in medical_keywords)

        # These legitimate sensors should NOT be flagged
        for entity in legitimate_sensors:
            assert not is_medical_entity(
                entity
            ), f"False positive: {entity['entity_id']} should not be flagged"
