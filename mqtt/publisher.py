"""
Cognitive IoT - MQTT Sensor Publisher

Generates simulated sensor readings and continuously
publishes them to the MQTT sensor topic. Supports controlled
demonstration scenarios for PBL presentation.
"""

import argparse
import json
import random
import time

import paho.mqtt.client as mqtt


# ============================================================
# MQTT CONFIGURATION
# ============================================================

BROKER = "localhost"
PORT = 1883

SENSOR_TOPIC = "cognitive-iot/sensors"
ACTUATOR_TOPIC = "cognitive-iot/actuators"


# ============================================================
# SENSOR SIMULATION & SCENARIOS
# ============================================================

def generate_sensor_data(scenario=None):
    """
    Generate sensor readings based on the selected demonstration scenario
    or default random mode.
    """
    if scenario == "empty":
        return {
            "temperature": round(random.uniform(20.0, 23.0), 2),
            "humidity": round(random.uniform(40.0, 55.0), 2),
            "light": round(random.uniform(50.0, 150.0), 2),
            "sound": round(random.uniform(0.05, 0.15), 2),
            "co2": round(random.uniform(400.0, 500.0), 2),
            "motion": 0.0,
        }
    elif scenario == "high_occupancy":
        return {
            "temperature": round(random.uniform(28.0, 30.0), 2),
            "humidity": round(random.uniform(55.0, 75.0), 2),
            "light": round(random.uniform(400.0, 750.0), 2),
            "sound": round(random.uniform(0.45, 0.75), 2),
            "co2": round(random.uniform(850.0, 1100.0), 2),
            "motion": 1.0,
        }
    elif scenario == "critical_co2":
        return {
            "temperature": round(random.uniform(25.0, 28.0), 2),
            "humidity": round(random.uniform(45.0, 65.0), 2),
            "light": round(random.uniform(300.0, 600.0), 2),
            "sound": round(random.uniform(0.20, 0.45), 2),
            "co2": round(random.uniform(1200.0, 1500.0), 2),
            "motion": 1.0,
        }
    else:
        # Default random mode
        return {
            "temperature": round(random.uniform(20.0, 30.0), 2),
            "humidity": round(random.uniform(40.0, 80.0), 2),
            "light": round(random.uniform(50.0, 800.0), 2),
            "sound": round(random.uniform(0.05, 0.80), 2),
            "co2": round(random.uniform(400.0, 1200.0), 2),
            "motion": round(random.choice([0.0, 0.0, 1.0]), 2),
        }


# ============================================================
# MQTT CALLBACK
# ============================================================

def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties,
):

    if reason_code == 0:

        print(
            "MQTT Publisher connected to broker."
        )

    else:

        print(
            f"MQTT connection failed: "
            f"{reason_code}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cognitive IoT MQTT Sensor Publisher"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["empty", "high_occupancy", "critical_co2"],
        default=None,
        help="Demonstration scenario to run (empty, high_occupancy, critical_co2)"
    )
    args = parser.parse_args()

    client = mqtt.Client(
        callback_api_version=
        mqtt.CallbackAPIVersion.VERSION2
    )

    client.on_connect = on_connect

    print(
        f"Connecting to MQTT broker "
        f"{BROKER}:{PORT}..."
    )

    client.connect(
        BROKER,
        PORT,
        keepalive=60,
    )

    client.loop_start()

    time.sleep(1)

    print(
        f"\nPublishing sensor data to:"
        f"\n{SENSOR_TOPIC}"
    )
    if args.scenario:
        print(f"Active Demonstration Scenario: {args.scenario}")
    else:
        print("Active Mode: Default (Random)")

    print(
        "\nPress Ctrl+C to stop.\n"
    )

    try:

        while True:

            sensor_data = (
                generate_sensor_data(scenario=args.scenario)
            )

            payload = json.dumps(
                sensor_data
            )

            result = client.publish(
                SENSOR_TOPIC,
                payload,
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:

                print(
                    "Published sensor data:"
                )

                print(
                    json.dumps(
                        sensor_data,
                        indent=2,
                    )
                )

            else:

                print(
                    f"Publish failed. "
                    f"MQTT error: {result.rc}"
                )

            print("-" * 50)

            time.sleep(5)

    except KeyboardInterrupt:

        print(
            "\nStopping MQTT publisher..."
        )

    finally:

        client.loop_stop()
        client.disconnect()

        print(
            "MQTT publisher disconnected."
        )


if __name__ == "__main__":
    main()
