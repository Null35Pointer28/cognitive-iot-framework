"""
Cognitive IoT - Dashboard MQTT Client

Maintains an MQTT connection for the Streamlit dashboard
and stores the latest sensor and actuator messages safely in memory.
"""

import json
import threading

import paho.mqtt.client as mqtt


BROKER = "localhost"
PORT = 1883

SENSOR_TOPIC = "cognitive-iot/sensors"
ACTUATOR_TOPIC = "cognitive-iot/actuators"


class DashboardMQTTClient:
    """
    Singleton MQTT client designed for the Streamlit dashboard.
    Runs communication in a background thread and maintains thread-safe
    access to the latest sensor and actuator messages.
    """

    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls, broker=BROKER, port=PORT):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, broker=BROKER, port=PORT):
        with self._singleton_lock:
            if getattr(self, "_initialized", False):
                return

            self.broker = broker
            self.port = port
            self.sensor_topic = SENSOR_TOPIC
            self.actuator_topic = ACTUATOR_TOPIC

            self.connected = False
            self.latest_sensor_data = None
            self.latest_actuator_data = None

            self.lock = threading.Lock()
            self._started = False

            try:
                self.client = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
                )
            except (AttributeError, TypeError):
                self.client = mqtt.Client()

            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect

            self._initialized = True

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
                client.subscribe(self.actuator_topic)
            except Exception:
                pass

    def _on_message(self, client, userdata, message):
        try:
            if message is None or message.payload is None:
                return

            payload_bytes = message.payload
            if isinstance(payload_bytes, bytes):
                payload_str = payload_bytes.decode("utf-8", errors="replace")
            else:
                payload_str = str(payload_bytes)

            payload_data = json.loads(payload_str)
            if not isinstance(payload_data, dict):
                return

            with self.lock:
                if message.topic == self.sensor_topic:
                    self.latest_sensor_data = payload_data
                elif message.topic == self.actuator_topic:
                    self.latest_actuator_data = payload_data
        except Exception:
            # Safely handle and ignore malformed messages without crashing
            pass

    def _on_disconnect(self, *args, **kwargs):
        with self.lock:
            self.connected = False

    def start(self):
        with self.lock:
            if self._started or self.connected:
                return
            self._started = True

        try:
            self.client.connect(
                self.broker,
                self.port,
                keepalive=60,
            )
            self.client.loop_start()
        except Exception:
            with self.lock:
                self.connected = False
                self._started = False

    def stop(self):
        with self.lock:
            self._started = False
            self.connected = False

        try:
            self.client.loop_stop()
        except Exception:
            pass

        try:
            self.client.disconnect()
        except Exception:
            pass

    def get_sensor_data(self):
        with self.lock:
            if self.latest_sensor_data is None:
                return None
            return dict(self.latest_sensor_data)

    def get_actuator_data(self):
        with self.lock:
            if self.latest_actuator_data is None:
                return None
            return dict(self.latest_actuator_data)

    def is_connected(self):
        with self.lock:
            return self.connected


# Module-level convenience functions delegating to the singleton client instance

def get_client() -> DashboardMQTTClient:
    return DashboardMQTTClient()


def start():
    get_client().start()


def stop():
    get_client().stop()


def get_sensor_data():
    return get_client().get_sensor_data()


def get_actuator_data():
    return get_client().get_actuator_data()


def is_connected():
    return get_client().is_connected()


if __name__ == "__main__":
    import time

    print("Running Dashboard MQTT Client standalone test...")
    client = DashboardMQTTClient()
    client.start()

    time.sleep(1.0)
    print(f"Connected to broker: {client.is_connected()}")
    print(f"Sensor data: {client.get_sensor_data()}")
    print(f"Actuator data: {client.get_actuator_data()}")

    print("Listening for messages for 3 seconds...")
    time.sleep(3.0)

    print(f"Latest Sensor data: {client.get_sensor_data()}")
    print(f"Latest Actuator data: {client.get_actuator_data()}")

    print("Stopping client...")
    client.stop()
    print("Test completed.")
