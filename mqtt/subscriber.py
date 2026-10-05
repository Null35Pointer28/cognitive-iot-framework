import json
import time
import paho.mqtt.client as mqtt

from ai.predict import OccupancyPredictor
from decision.decision_engine import CognitiveDecisionEngine
from actuators.virtual_actuators import VirtualActuatorSystem
from feedback.feedback_manager import FeedbackManager


BROKER = "localhost"
PORT = 1883

SENSOR_TOPIC = "cognitive-iot/sensors"
ACTUATOR_TOPIC = "cognitive-iot/actuators"


class CognitiveIoTSubscriber:

    def __init__(self):

        self.predictor = OccupancyPredictor()
        self.decision_engine = CognitiveDecisionEngine()
        self.actuators = VirtualActuatorSystem()
        self.feedback_manager = FeedbackManager()

        self.client = mqtt.Client()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):

        if rc == 0:
            print(f"\nConnected to MQTT broker: {BROKER}:{PORT}")

            client.subscribe(SENSOR_TOPIC)

            print(f"Subscribed to: {SENSOR_TOPIC}")
            print("\nWaiting for sensor data...\n")

        else:
            print(
                f"MQTT connection failed with code: {rc}"
            )

    def on_message(self, client, userdata, msg):

        try:

            print("\n" + "=" * 60)
            print("NEW SENSOR READING")
            print("=" * 60)

            # =================================================
            # 1. SENSOR DATA
            # =================================================

            sensor_data = json.loads(
                msg.payload.decode()
            )

            temperature = float(
                sensor_data["temperature"]
            )

            humidity = float(
                sensor_data["humidity"]
            )

            light = float(
                sensor_data["light"]
            )

            sound = float(
                sensor_data["sound"]
            )

            co2 = float(
                sensor_data["co2"]
            )

            motion = float(
                sensor_data["motion"]
            )

            print("\n[1] SENSOR DATA")

            print(
                f"Temperature : {temperature:.2f} °C"
            )

            print(
                f"Humidity    : {humidity:.2f} %"
            )

            print(
                f"Light       : {light:.2f}"
            )

            print(
                f"Sound       : {sound:.2f}"
            )

            print(
                f"CO2         : {co2:.2f} ppm"
            )

            print(
                f"Motion      : {motion:.2f}"
            )

            # =================================================
            # 2. AI UNDERSTANDING
            # =================================================

            prediction = self.predictor.predict(
                sensor_data
            )

            print("\n[2] AI UNDERSTANDING")

            print(
                f"Occupancy Class : "
                f"{prediction['occupancy_class']}"
            )

            print(
                f"Occupancy State : "
                f"{prediction['occupancy_state']}"
            )

            print(
                f"Confidence      : "
                f"{prediction['confidence']:.2%}"
            )

            # =================================================
            # 3. COGNITIVE DECISION
            # =================================================

            decision = (
                self.decision_engine.make_decision(
                    prediction,
                    temperature,
                    co2,
                    light,
                    sound
                )
            )

            print("\n[3] COGNITIVE DECISION")

            print(
                f"Cooling      : "
                f"{decision.cooling}"
            )

            print(
                f"Ventilation  : "
                f"{decision.ventilation}"
            )

            print(
                f"Lighting     : "
                f"{decision.lighting}"
            )

            print(
                f"Priority     : "
                f"{decision.priority}"
            )

            # =================================================
            # 4. ACTUATOR RESPONSE
            # =================================================

            print("\n[4] ACTUATOR RESPONSE")

            self.actuators.apply_decision(
                decision
            )

            self.actuators.display_state()

            # =================================================
            # 5. MQTT ACTUATOR COMMAND
            # =================================================

            actuator_command = {
                "cooling": decision.cooling,
                "ventilation": decision.ventilation,
                "lighting": decision.lighting,
                "occupancy_state": (
                    decision.occupancy_state
                ),
                "confidence": (
                    decision.occupancy_confidence
                ),
                "priority": decision.priority
            }

            actuator_payload = json.dumps(
                actuator_command
            )

            self.client.publish(
                ACTUATOR_TOPIC,
                actuator_payload
            )

            print("\n[5] MQTT ACTUATOR COMMAND")

            print(
                f"Published → {ACTUATOR_TOPIC}"
            )

            print(
                json.dumps(
                    actuator_command,
                    indent=2
                )
            )

            # =================================================
            # 6. FEEDBACK RECORDING
            # =================================================

            try:

                self.feedback_manager.record_feedback(
                    sensor_data,
                    prediction,
                    decision
                )

                print("\n[6] FEEDBACK")

                print(
                    "Feedback record stored successfully."
                )

            except Exception as feedback_error:

                print(
                    "\nFeedback recording failed: "
                    f"{feedback_error}"
                )

            print("\n" + "=" * 60)

        except Exception as error:

            print(
                "\nERROR while processing "
                f"sensor data: {error}"
            )

    def start(self):

        print("=" * 60)
        print("       COGNITIVE IoT MQTT SUBSCRIBER")
        print("=" * 60)

        try:

            print(
                f"\nConnecting to MQTT broker "
                f"{BROKER}:{PORT}..."
            )

            self.client.connect(
                BROKER,
                PORT,
                60
            )

            self.client.loop_start()

            while True:

                time.sleep(1)

        except KeyboardInterrupt:

            print(
                "\n\nStopping MQTT subscriber..."
            )

        except Exception as error:

            print(
                f"\nSubscriber error: {error}"
            )

        finally:

            self.client.loop_stop()

            self.client.disconnect()

            print(
                "Subscriber stopped."
            )


if __name__ == "__main__":

    subscriber = CognitiveIoTSubscriber()

    subscriber.start()