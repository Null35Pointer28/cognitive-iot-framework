"""
Cognitive IoT - MQTT Cognitive Bridge

Integrates incoming Virtual ESP32 sensor telemetry with the CognitivePipeline:
- Subscribes to cognitive-iot/sensors
- Passes raw sensor payloads to CognitivePipeline.process()
- Publishes resulting cognitive actuator commands to cognitive-iot/actuators
- Maintains thread-safe in-memory state and statistics.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import threading
import time
from datetime import datetime

import paho.mqtt.client as mqtt

from cognitive.pipeline import CognitivePipeline
from config.config import MQTT


class CognitiveMQTTBridge:
    """
    Bridges MQTT sensor telemetry and actuator commands with the CognitivePipeline.
    """

    def __init__(
        self,
        broker: str = MQTT.get("broker", "localhost"),
        port: int = MQTT.get("port", 1883),
        record_feedback: bool = True,
    ):
        self.broker = broker
        self.port = port

        topics = MQTT.get("topics", {})
        self.sensor_topic = topics.get("sensors", "cognitive-iot/sensors")
        self.actuator_topic = topics.get("actuators", "cognitive-iot/actuators")

        self.pipeline = CognitivePipeline(record_feedback=record_feedback)

        self.connected = False
        self.running = False

        self.latest_sensor_data = None
        self.latest_cognitive_result = None
        self.latest_actuator_command = None
        self.message_count = 0
        self.last_message_timestamp = None

        self.lock = threading.Lock()

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
                client.subscribe(self.sensor_topic)
            except Exception:
                pass

    def _on_message(self, client, userdata, message):
        try:
            if message is None or message.payload is None:
                return

            payload_str = message.payload.decode("utf-8", errors="replace")
            sensor_data = json.loads(payload_str)

            if not isinstance(sensor_data, dict):
                return

            # Process through CognitivePipeline
            result = self.pipeline.process(sensor_data)

            prediction = result.get("prediction", {})
            decision = result.get("decision", {})

            actuator_command = {
                "cooling": decision.get("cooling", "OFF"),
                "ventilation": decision.get("ventilation", "OFF"),
                "lighting": decision.get("lighting", "OFF"),
                "occupancy_state": prediction.get("occupancy_state", "Unknown"),
                "confidence": prediction.get("confidence", 0.0),
                "priority": decision.get("priority", "NORMAL"),
            }

            payload = json.dumps(actuator_command)
            self.client.publish(self.actuator_topic, payload)

            with self.lock:
                self.latest_sensor_data = sensor_data
                self.latest_cognitive_result = result
                self.latest_actuator_command = actuator_command
                self.message_count += 1
                self.last_message_timestamp = datetime.now().isoformat(timespec="seconds")

        except Exception:
            # Gracefully handle malformed messages without crashing
            pass

    def _on_disconnect(self, *args, **kwargs):
        with self.lock:
            self.connected = False

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True

        try:
            self.client.connect(
                self.broker,
                self.port,
                keepalive=MQTT.get("keepalive", 60),
            )
            self.client.loop_start()
        except Exception:
            with self.lock:
                self.connected = False
                self.running = False

    def stop(self):
        with self.lock:
            self.running = False

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

    def get_status(self) -> dict:
        with self.lock:
            return {
                "connected": self.connected,
                "running": self.running,
                "message_count": self.message_count,
                "last_message_timestamp": self.last_message_timestamp,
            }

    def get_latest_sensor_data(self) -> dict:
        with self.lock:
            if self.latest_sensor_data is None:
                return {}
            return dict(self.latest_sensor_data)

    def get_latest_cognitive_result(self) -> dict:
        with self.lock:
            if self.latest_cognitive_result is None:
                return {}
            return dict(self.latest_cognitive_result)

    def get_latest_actuator_command(self) -> dict:
        with self.lock:
            if self.latest_actuator_command is None:
                return {}
            return dict(self.latest_actuator_command)


if __name__ == "__main__":
    print("========================================")
    print("     COGNITIVE MQTT BRIDGE INTEGRATION TEST")
    print("========================================")

    from simulation.virtual_esp32 import VirtualESP32

    # 1. Start Bridge
    bridge = CognitiveMQTTBridge(record_feedback=False)
    bridge.start()
    time.sleep(1.0)
    print(f"Bridge status after start: {bridge.get_status()}")

    # 2. Start Virtual ESP32
    esp32 = VirtualESP32(scenario="high_occupancy", publish_interval_seconds=1)
    esp32.start()
    time.sleep(1.0)

    # Test scenarios: empty, high_occupancy, critical_co2
    scenarios_to_test = ["empty", "high_occupancy", "critical_co2"]

    for scn in scenarios_to_test:
        print(f"\n--- Testing Scenario: {scn} ---")
        esp32.set_scenario(scn)
        time.sleep(2.5)  # allow messages to flow through MQTT -> Bridge -> Pipeline -> MQTT -> ESP32

        status = bridge.get_status()
        sensor_data = bridge.get_latest_sensor_data()
        cognitive_result = bridge.get_latest_cognitive_result()
        actuator_cmd = bridge.get_latest_actuator_command()
        esp32_state = esp32.get_actuator_state()

        print(f"Messages processed so far : {status['message_count']}")
        print(f"Latest Sensor Readings  : {sensor_data}")
        print(f"AI Prediction           : {cognitive_result.get('prediction')}")
        print(f"Cognitive Decision      : {cognitive_result.get('decision')}")
        print(f"Published Actuator Cmd  : {actuator_cmd}")
        print(f"Virtual ESP32 Actuators : {esp32_state}")

    # 3. Clean Shutdown
    print("\nShutting down Virtual ESP32 and Bridge cleanly...")
    esp32.stop()
    bridge.stop()
    print("Integration test completed successfully.")
