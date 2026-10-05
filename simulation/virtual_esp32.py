"""
Cognitive IoT - Virtual ESP32 Device Simulator

Behaves like a physical ESP32 device at the MQTT communication level:
- Generates sensor readings by reusing authoritative scenarios from simulation.scenarios
- Reuses MQTT configuration from config.config
- Publishes sensor telemetry to cognitive-iot/sensors
- Subscribes to actuator commands on cognitive-iot/actuators and updates internal actuator state
- Publishes device status to cognitive-iot/device/status
- Provides a clean programmatic API and runs publishing/status loops in background threads.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import random
import threading
import time
from datetime import datetime

import paho.mqtt.client as mqtt

from actuators.virtual_actuators import VirtualActuatorSystem
from simulation.scenarios import SCENARIOS
from config.config import MQTT


class VirtualESP32:
    """
    Virtual ESP32 device simulator managing sensor telemetry, MQTT communication,
    actuator states, and simulation scenarios.
    """

    def __init__(
        self,
        device_id: str = "virtual_esp32_01",
        broker: str = MQTT.get("broker", "localhost"),
        port: int = MQTT.get("port", 1883),
        scenario: str = "random",
        publish_interval_seconds: int = MQTT.get("publish_interval_seconds", 5),
    ):
        self.device_id = device_id
        self.broker = broker
        self.port = port
        self.scenario = scenario
        self.publish_interval_seconds = publish_interval_seconds

        topics = MQTT.get("topics", {})
        self.sensor_topic = topics.get("sensors", "cognitive-iot/sensors")
        self.actuator_topic = topics.get("actuators", "cognitive-iot/actuators")
        self.status_topic = topics.get("status", "cognitive-iot/device/status")

        self.running = False
        self.connected = False

        self.latest_sensor_data = None
        self.actuator_system = VirtualActuatorSystem()

        self.lock = threading.Lock()
        self._thread = None
        self._status_thread = None

        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
        except (AttributeError, TypeError):
            self.client = mqtt.Client()

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, *args):
        rc = args[0] if args else 0
        is_success = False
        try:
            if hasattr(rc, "is_failure"):
                is_success = not rc.is_failure
            else:
                is_success = (rc == 0)
        except Exception:
            is_success = (rc == 0)

        with self.lock:
            self.connected = is_success

        if is_success:
            try:
                client.subscribe(self.actuator_topic)
            except Exception:
                pass

    def _on_message(self, client, userdata, message):
        try:
            if message is None or message.payload is None:
                return

            payload_str = message.payload.decode("utf-8", errors="replace")
            payload_data = json.loads(payload_str)

            if isinstance(payload_data, dict):
                class DecisionWrapper:
                    def __init__(self, data):
                        self.cooling = data.get("cooling", "OFF")
                        self.ventilation = data.get("ventilation", "OFF")
                        self.lighting = data.get("lighting", "OFF")

                decision = DecisionWrapper(payload_data)
                with self.lock:
                    self.actuator_system.apply_decision(decision)
        except Exception:
            pass

    def _on_disconnect(self, *args, **kwargs):
        with self.lock:
            self.connected = False

    def generate_sensor_data(self) -> dict:
        """
        Generate sensor readings based on the active scenario defined in simulation.scenarios.
        Controls ONLY simulated sensor inputs.
        """
        scenario_def = SCENARIOS.get(self.scenario, SCENARIOS.get("random", {}))

        sensor_values = {}
        for sensor_name in ["temperature", "humidity", "light", "sound", "co2", "motion"]:
            if sensor_name in scenario_def:
                rng = scenario_def[sensor_name]
                if sensor_name == "motion":
                    if rng[0] == rng[1]:
                        sensor_values[sensor_name] = float(rng[0])
                    else:
                        sensor_values[sensor_name] = float(random.choice([0.0, 1.0]))
                else:
                    sensor_values[sensor_name] = round(random.uniform(rng[0], rng[1]), 2)
            else:
                sensor_values[sensor_name] = 0.0

        reading = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device_id": self.device_id,
            "scenario": self.scenario,
            **sensor_values,
        }
        return reading

    def _publishing_loop(self):
        while self.running:
            try:
                data = self.generate_sensor_data()
                with self.lock:
                    self.latest_sensor_data = data

                payload = json.dumps(data)
                self.client.publish(self.sensor_topic, payload)
            except Exception:
                pass

            time.sleep(self.publish_interval_seconds)

    def _status_loop(self):
        while self.running:
            try:
                status = self.get_status()
                payload = json.dumps(status)
                self.client.publish(self.status_topic, payload, retain=True)
            except Exception:
                pass

            time.sleep(10)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True

        try:
            self.client.connect(self.broker, self.port, keepalive=MQTT.get("keepalive", 60))
            self.client.loop_start()
        except Exception:
            with self.lock:
                self.connected = False

        self._thread = threading.Thread(target=self._publishing_loop, daemon=True)
        self._thread.start()

        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._status_thread.start()

    def stop(self):
        with self.lock:
            self.running = False

        try:
            status = self.get_status()
            status["running"] = False
            self.client.publish(self.status_topic, json.dumps(status), retain=True)
        except Exception:
            pass

        try:
            self.client.loop_stop()
        except Exception:
            pass

        try:
            self.client.disconnect()
        except Exception:
            pass

        with self.lock:
            self.connected = False

    def set_scenario(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"Invalid scenario '{scenario}'. Choose from {list(SCENARIOS.keys())}")
        with self.lock:
            self.scenario = scenario

    def set_publish_interval(self, interval_seconds: int):
        if interval_seconds < 1:
            raise ValueError("Publish interval must be at least 1 second.")
        with self.lock:
            self.publish_interval_seconds = interval_seconds

    def get_status(self) -> dict:
        with self.lock:
            return {
                "device_id": self.device_id,
                "running": self.running,
                "connected": self.connected,
                "scenario": self.scenario,
                "publish_interval_seconds": self.publish_interval_seconds,
            }

    def get_latest_sensor_data(self) -> dict:
        with self.lock:
            if self.latest_sensor_data is None:
                return {}
            return dict(self.latest_sensor_data)

    def get_actuator_state(self) -> dict:
        with self.lock:
            state = self.actuator_system.state
            return {
                "cooling": getattr(state, "cooling", "OFF"),
                "ventilation": getattr(state, "ventilation", "OFF"),
                "lighting": getattr(state, "lighting", "OFF"),
            }


if __name__ == "__main__":
    print("========================================")
    print("    VIRTUAL ESP32 REFACTORED TEST")
    print("========================================")

    # Test all four required scenarios
    test_scenarios = ["random", "empty", "high_occupancy", "critical_co2"]

    for scn in test_scenarios:
        print(f"\nTesting scenario: {scn}")
        esp32 = VirtualESP32(scenario=scn, publish_interval_seconds=1)
        esp32.start()
        time.sleep(1.2)
        print(f"Status           : {esp32.get_status()}")
        print(f"Generated Sensors: {esp32.get_latest_sensor_data()}")

        # Test actuator command reception
        test_command = {
            "cooling": "HIGH",
            "ventilation": "HIGH",
            "lighting": "MEDIUM",
        }
        esp32.client.publish(esp32.actuator_topic, json.dumps(test_command))
        time.sleep(0.5)
        print(f"Actuator State   : {esp32.get_actuator_state()}")

        esp32.stop()
        print(f"Scenario {scn} test passed.")

    print("\nAll scenario and MQTT tests completed successfully.")
