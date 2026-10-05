"""
Central configuration for the Cognitive IoT project.

This file contains the sensor definitions, realistic value ranges,
MQTT settings, and general project settings used throughout the application.
"""

# ============================================================
# SENSOR CONFIGURATION
# ============================================================

SENSORS = {
    "temperature": {
        "unit": "°C",
        "min": 15.0,
        "max": 40.0,
        "normal_min": 20.0,
        "normal_max": 30.0,
    },

    "humidity": {
        "unit": "%",
        "min": 20.0,
        "max": 90.0,
        "normal_min": 40.0,
        "normal_max": 65.0,
    },

    "light": {
        "unit": "lux",
        "min": 0.0,
        "max": 1000.0,
        "normal_min": 200.0,
        "normal_max": 800.0,
    },

    "noise": {
        "unit": "dB",
        "min": 20.0,
        "max": 100.0,
        "normal_min": 20.0,
        "normal_max": 60.0,
    },

    "presence": {
        "unit": "boolean",
        "min": 0,
        "max": 1,
    },

    "air_quality": {
        "unit": "index",
        "min": 0.0,
        "max": 100.0,
        "normal_min": 60.0,
        "normal_max": 100.0,
    },
}


# ============================================================
# MQTT CONFIGURATION
# ============================================================

MQTT = {
    "broker": "localhost",
    "port": 1883,
    "topics": {
        "sensors": "cognitive-iot/sensors",
        "actuators": "cognitive-iot/actuators",
        "status": "cognitive-iot/device/status",
    },
    "keepalive": 60,
    "publish_interval_seconds": 5,
}


# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

SIMULATION = {
    "default_samples": 1000,
    "interval_seconds": 1,
    "random_seed": 42,
}


# ============================================================
# ENVIRONMENT SCENARIOS
# ============================================================

SCENARIOS = [
    "random",
    "empty",
    "high_occupancy",
    "critical_co2",
    "comfortable",
    "hot",
    "poor_lighting",
    "noisy",
    "poor_air_quality",
    "empty_room",
    "multiple_issues",
]


# ============================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================

ML = {
    "test_size": 0.2,
    "random_state": 42,
}
