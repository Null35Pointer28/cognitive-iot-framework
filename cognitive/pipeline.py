"""
Cognitive IoT - Reusable Cognitive Pipeline

Encapsulates the core cognitive processing flow:
Sensor Data → OccupancyPredictor → CognitiveDecisionEngine → VirtualActuatorSystem → Cognitive Result

Does NOT handle MQTT communication, background threads, or networking.
Designed to be called synchronously by the MQTT subscriber / integration layer or FastAPI backend.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import threading
from datetime import datetime

from ai.predict import OccupancyPredictor
from decision.decision_engine import CognitiveDecisionEngine
from actuators.virtual_actuators import VirtualActuatorSystem
from feedback.feedback_manager import FeedbackManager


class CognitivePipeline:
    """
    Core cognitive processing pipeline for the Cognitive IoT Framework.
    """

    def __init__(self, record_feedback: bool = True):
        self.predictor = OccupancyPredictor()
        self.decision_engine = CognitiveDecisionEngine()
        self.actuators = VirtualActuatorSystem()
        self.feedback_manager = FeedbackManager() if record_feedback else None

        self.latest_result = None
        self.lock = threading.Lock()

    def process(self, sensor_data: dict) -> dict:
        """
        Process incoming sensor data through AI prediction, cognitive decision making,
        and virtual actuators, returning a serializable cognitive result.
        """
        if not isinstance(sensor_data, dict):
            raise ValueError("Sensor data must be a dictionary.")

        # 1. Validate incoming sensor data with safe defaults / extraction
        try:
            temperature = float(sensor_data.get("temperature", 22.0))
            humidity = float(sensor_data.get("humidity", 50.0))
            light = float(sensor_data.get("light", 300.0))
            sound = float(sensor_data.get("sound", 0.2))
            co2 = float(sensor_data.get("co2", 500.0))
            motion = float(sensor_data.get("motion", 0.0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid sensor data types: {e}") from e

        # Normalize keys expected by predictor
        normalized_sensor = {
            "temperature": temperature,
            "humidity": humidity,
            "light": light,
            "sound": sound,
            "co2": co2,
            "motion": motion,
        }

        # 2. AI Prediction
        prediction = self.predictor.predict(normalized_sensor)

        # 3. Cognitive Decision
        decision = self.decision_engine.make_decision(
            prediction=prediction,
            temperature=temperature,
            co2=co2,
            light=light,
            sound=sound,
        )

        # 4. Apply to Virtual Actuators
        actuator_state = self.actuators.apply_decision(decision)

        # 5. Optional Feedback Recording (recording pending decision)
        if self.feedback_manager is not None:
            try:
                self.feedback_manager.record_feedback(
                    sensor_data=normalized_sensor,
                    prediction=prediction,
                    decision=decision,
                )
            except Exception:
                pass

        # 6. Construct Serializable Result
        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prediction": {
                "occupancy_class": int(prediction.get("occupancy_class", 0)),
                "occupancy_state": str(prediction.get("occupancy_state", "Unknown")),
                "confidence": float(prediction.get("confidence", 0.0)),
            },
            "decision": {
                "cooling": str(decision.cooling),
                "ventilation": str(decision.ventilation),
                "lighting": str(decision.lighting),
                "priority": str(decision.priority),
                "reasons": list(decision.reasons),
            },
            "actuators": {
                "cooling": str(getattr(actuator_state, "cooling", "OFF")),
                "ventilation": str(getattr(actuator_state, "ventilation", "OFF")),
                "lighting": str(getattr(actuator_state, "lighting", "OFF")),
            },
        }

        # Thread-safe update of latest result
        with self.lock:
            self.latest_result = result

        return result

    def get_latest_result(self) -> dict:
        """
        Retrieve the most recent cognitive processing result thread-safely.
        """
        with self.lock:
            if self.latest_result is None:
                return {}
            return dict(self.latest_result)


if __name__ == "__main__":
    print("========================================")
    print("     COGNITIVE PIPELINE STANDALONE TEST")
    print("========================================")

    pipeline = CognitivePipeline(record_feedback=False)

    test_sensor_data = {
        "temperature": 28.5,
        "humidity": 65.0,
        "light": 500.0,
        "sound": 0.55,
        "co2": 950.0,
        "motion": 1.0,
    }

    print("\nInput Sensor Data:")
    print(json.dumps(test_sensor_data, indent=2))

    # Process through pipeline
    result = pipeline.process(test_sensor_data)

    print("\nCognitive Pipeline Result:")
    print(json.dumps(result, indent=2))

    # Verify JSON serializability
    serialized = json.dumps(result)
    print(f"\nJSON Serialization check: Success ({len(serialized)} bytes)")

    # Verify get_latest_result()
    latest = pipeline.get_latest_result()
    print(f"get_latest_result() check: Success (matches result: {latest == result})")

    # Verify virtual actuator state changes
    actuator_state = pipeline.actuators.state
    print(f"Virtual Actuator State: Cooling={actuator_state.cooling}, Ventilation={actuator_state.ventilation}, Lighting={actuator_state.lighting}")

    print("\nAll pipeline tests completed successfully without requiring MQTT.")
