"""
Cognitive IoT - Real-Time Cognitive Pipeline

Complete pipeline:

Sensors
   ↓
Sensor Fusion
   ↓
AI Understanding
   ↓
Cognitive Decision
   ↓
Virtual Actuators
"""

import random
import time

from ai.predict import OccupancyPredictor
from decision.decision_engine import CognitiveDecisionEngine
from actuators.virtual_actuators import VirtualActuatorSystem


SENSOR_RANGES = {
    "temperature": (20.0, 32.0),
    "light": (20.0, 800.0),
    "sound": (0.02, 0.80),
    "co2": (400.0, 1400.0),
    "motion": (0.0, 1.0),
}


def generate_sensor_reading():
    """
    Generate simulated real-time sensor readings.
    """

    return {
        "temperature": round(
            random.uniform(*SENSOR_RANGES["temperature"]), 2
        ),
        "light": round(
            random.uniform(*SENSOR_RANGES["light"]), 2
        ),
        "sound": round(
            random.uniform(*SENSOR_RANGES["sound"]), 2
        ),
        "co2": round(
            random.uniform(*SENSOR_RANGES["co2"]), 2
        ),
        "motion": round(
            random.uniform(*SENSOR_RANGES["motion"]), 2
        ),
    }


def convert_to_fused_data(reading):
    """
    Convert sensor readings into the feature format
    expected by the trained AI model.
    """

    temperature_score = (
        (reading["temperature"] - 20.0) / 12.0
    )

    lighting_score = (
        (reading["light"] - 20.0) / 780.0
    )

    sound_score = (
        (reading["sound"] - 0.02) / 0.78
    )

    co2_score = (
        (reading["co2"] - 400.0) / 1000.0
    )

    motion_score = reading["motion"]

    unified_context_score = (
        temperature_score
        + lighting_score
        + sound_score
        + co2_score
        + motion_score
    ) / 5

    return {
        "temperature_context": reading["temperature"],
        "lighting_context": reading["light"],
        "sound_context": reading["sound"],
        "co2_context": reading["co2"],
        "motion_context": reading["motion"],
        "unified_context_score": unified_context_score,
    }


def display_result(
    sensor_data,
    prediction,
    decision,
):
    """
    Display the complete cognitive processing result.
    """

    print("\n" + "=" * 75)
    print("COGNITIVE IoT - REAL-TIME COGNITIVE PIPELINE")
    print("=" * 75)

    # --------------------------------------------------
    # SENSOR INPUT
    # --------------------------------------------------

    print("\n[1] SENSOR INPUT")
    print("-" * 45)

    for sensor, value in sensor_data.items():
        print(f"{sensor:<15} {value}")

    # --------------------------------------------------
    # AI UNDERSTANDING
    # --------------------------------------------------

    print("\n[2] AI UNDERSTANDING")
    print("-" * 45)

    print(
        f"Occupancy State : "
        f"{prediction['occupancy_state']}"
    )

    print(
        f"Confidence      : "
        f"{prediction['confidence'] * 100:.2f}%"
    )

    # --------------------------------------------------
    # COGNITIVE DECISION
    # --------------------------------------------------

    print("\n[3] COGNITIVE DECISION")
    print("-" * 45)

    print(f"Cooling         : {decision.cooling}")
    print(f"Ventilation     : {decision.ventilation}")
    print(f"Lighting        : {decision.lighting}")
    print(f"Priority        : {decision.priority}")

    print("\nDecision Reasons")

    for reason in decision.reasons:
        print(f"- {reason}")


def run_pipeline(interval=3):
    """
    Run the complete real-time Cognitive IoT pipeline.
    """

    # AI component
    predictor = OccupancyPredictor()

    # Decision component
    decision_engine = CognitiveDecisionEngine()

    # Actuator component
    actuator_system = VirtualActuatorSystem()

    print("=" * 75)
    print("COGNITIVE IoT - REAL-TIME COGNITIVE PIPELINE")
    print("=" * 75)

    print("\nStarting real-time cognitive processing...")
    print("Press Ctrl+C to stop.\n")

    try:

        while True:

            # ==================================================
            # 1. SENSE
            # ==================================================

            sensor_data = generate_sensor_reading()

            # ==================================================
            # 2. UNDERSTAND
            # ==================================================

            fused_data = convert_to_fused_data(
                sensor_data
            )

            prediction = predictor.predict(
                fused_data
            )

            # ==================================================
            # 3. DECIDE
            # ==================================================

            decision = decision_engine.make_decision(
                prediction=prediction,
                temperature=sensor_data["temperature"],
                co2=sensor_data["co2"],
                light=sensor_data["light"],
                sound=sensor_data["sound"],
            )

            # ==================================================
            # 4. ACT
            # ==================================================

            actuator_system.apply_decision(
                decision
            )

            # ==================================================
            # DISPLAY COMPLETE PIPELINE
            # ==================================================

            display_result(
                sensor_data,
                prediction,
                decision,
            )

            actuator_system.display_state()

            # ==================================================
            # WAIT FOR NEXT SENSOR CYCLE
            # ==================================================

            time.sleep(interval)

    except KeyboardInterrupt:

        print(
            "\n\nReal-time cognitive pipeline stopped."
        )


if __name__ == "__main__":
    run_pipeline()