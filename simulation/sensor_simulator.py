"""
Virtual sensor simulator for the Cognitive IoT project.

This module generates realistic sensor readings for different
environmental scenarios. It acts as a software replacement for
the physical sensors that we will integrate later.
"""

import random
from datetime import datetime

from simulation.scenarios import SCENARIOS


class SensorSimulator:
    """
    Generates virtual sensor readings based on an environmental scenario.
    """

    def __init__(self, scenario="comfortable"):
        """
        Initialize the simulator.

        Parameters:
            scenario (str): Name of the environmental scenario.
        """

        if scenario not in SCENARIOS:
            available = ", ".join(SCENARIOS.keys())
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Available scenarios: {available}"
            )

        self.scenario = scenario

    def set_scenario(self, scenario):
        """
        Change the current environmental scenario.
        """

        if scenario not in SCENARIOS:
            available = ", ".join(SCENARIOS.keys())
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Available scenarios: {available}"
            )

        self.scenario = scenario

    def _generate_value(self, sensor_name):
        """
        Generate a random value within the range defined
        for the current scenario.
        """

        sensor_range = SCENARIOS[self.scenario][sensor_name]

        minimum, maximum = sensor_range

        # Presence is a binary value.
        if sensor_name == "presence":
            return minimum

        # Generate a decimal value for continuous sensors.
        return round(random.uniform(minimum, maximum), 2)

    def read_sensors(self):
        """
        Generate one complete sensor reading.

        Returns:
            dict: Sensor readings with timestamp and scenario.
        """

        reading = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": self.scenario,
            "temperature": self._generate_value("temperature"),
            "humidity": self._generate_value("humidity"),
            "light": self._generate_value("light"),
            "noise": self._generate_value("noise"),
            "presence": self._generate_value("presence"),
            "air_quality": self._generate_value("air_quality"),
        }

        return reading

    def generate_samples(self, number_of_samples=10):
        """
        Generate multiple sensor readings.

        Parameters:
            number_of_samples (int): Number of readings to generate.

        Returns:
            list: List containing sensor readings.
        """

        samples = []

        for _ in range(number_of_samples):
            samples.append(self.read_sensors())

        return samples


if __name__ == "__main__":
    # Create a simulator for a hot environment.
    simulator = SensorSimulator("hot")

    # Generate and display five readings.
    readings = simulator.generate_samples(5)

    for reading in readings:
        print(reading)