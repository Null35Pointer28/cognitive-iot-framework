"""
Defines the environmental scenarios used by the Cognitive IoT simulator.

Each scenario contains realistic ranges for the virtual sensors.
The simulator and Virtual ESP32 use these ranges to generate sensor readings.
"""

SCENARIOS = {
    "random": {
        "description": "Default random indoor environment.",
        "temperature": (20.0, 30.0),
        "humidity": (40.0, 80.0),
        "light": (50.0, 800.0),
        "sound": (0.05, 0.80),
        "co2": (400.0, 1200.0),
        "motion": (0.0, 1.0),
        "air_quality": (60.0, 100.0),
    },

    "empty": {
        "description": "An unoccupied indoor environment.",
        "temperature": (20.0, 23.0),
        "humidity": (40.0, 55.0),
        "light": (50.0, 150.0),
        "sound": (0.05, 0.15),
        "co2": (400.0, 500.0),
        "motion": (0.0, 0.0),
        "air_quality": (75.0, 100.0),
    },

    "high_occupancy": {
        "description": "A hot, high occupancy environment.",
        "temperature": (28.0, 30.0),
        "humidity": (55.0, 75.0),
        "light": (400.0, 750.0),
        "sound": (0.45, 0.75),
        "co2": (850.0, 1100.0),
        "motion": (1.0, 1.0),
        "air_quality": (50.0, 75.0),
    },

    "critical_co2": {
        "description": "An environment with critical CO2 levels.",
        "temperature": (25.0, 28.0),
        "humidity": (45.0, 65.0),
        "light": (300.0, 600.0),
        "sound": (0.20, 0.45),
        "co2": (1200.0, 1500.0),
        "motion": (1.0, 1.0),
        "air_quality": (10.0, 45.0),
    },

    "comfortable": {
        "description": "A comfortable occupied indoor environment.",
        "temperature": (22.0, 27.0),
        "humidity": (40.0, 60.0),
        "light": (300.0, 800.0),
        "sound": (25.0, 45.0),
        "co2": (400.0, 800.0),
        "motion": (1.0, 1.0),
        "air_quality": (75.0, 100.0),
    },

    "hot": {
        "description": "A hot and humid occupied environment.",
        "temperature": (30.0, 38.0),
        "humidity": (65.0, 85.0),
        "light": (300.0, 800.0),
        "sound": (30.0, 55.0),
        "co2": (600.0, 1000.0),
        "motion": (1.0, 1.0),
        "air_quality": (70.0, 100.0),
    },

    "poor_lighting": {
        "description": "An occupied room with insufficient lighting.",
        "temperature": (22.0, 28.0),
        "humidity": (40.0, 65.0),
        "light": (20.0, 120.0),
        "sound": (25.0, 50.0),
        "co2": (500.0, 900.0),
        "motion": (1.0, 1.0),
        "air_quality": (70.0, 100.0),
    },

    "noisy": {
        "description": "An occupied environment with high noise levels.",
        "temperature": (22.0, 30.0),
        "humidity": (40.0, 65.0),
        "light": (300.0, 800.0),
        "sound": (65.0, 95.0),
        "co2": (500.0, 900.0),
        "motion": (1.0, 1.0),
        "air_quality": (70.0, 100.0),
    },

    "poor_air_quality": {
        "description": "An occupied environment with poor air quality.",
        "temperature": (22.0, 30.0),
        "humidity": (45.0, 70.0),
        "light": (300.0, 800.0),
        "sound": (25.0, 55.0),
        "co2": (1200.0, 1600.0),
        "motion": (1.0, 1.0),
        "air_quality": (10.0, 45.0),
    },

    "empty_room": {
        "description": "An unoccupied indoor environment.",
        "temperature": (20.0, 28.0),
        "humidity": (40.0, 65.0),
        "light": (50.0, 400.0),
        "sound": (20.0, 35.0),
        "co2": (400.0, 600.0),
        "motion": (0.0, 0.0),
        "air_quality": (75.0, 100.0),
    },

    "multiple_issues": {
        "description": "An occupied environment with multiple problems.",
        "temperature": (31.0, 38.0),
        "humidity": (65.0, 85.0),
        "light": (10.0, 100.0),
        "sound": (65.0, 95.0),
        "co2": (1200.0, 1600.0),
        "motion": (1.0, 1.0),
        "air_quality": (10.0, 45.0),
    },
}
