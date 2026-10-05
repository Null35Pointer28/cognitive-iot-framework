"""
Cognitive IoT - Virtual Actuator System

Simulates physical actuators before ESP32 hardware integration.
"""

from dataclasses import dataclass


@dataclass
class ActuatorState:
    cooling: str = "OFF"
    ventilation: str = "OFF"
    lighting: str = "OFF"


class VirtualActuatorSystem:

    def __init__(self):
        self.state = ActuatorState()

    def apply_decision(self, decision):
        """
        Apply the cognitive decision to the virtual actuators.
        """

        self.state.cooling = decision.cooling
        self.state.ventilation = decision.ventilation
        self.state.lighting = decision.lighting

        return self.state

    def display_state(self):
        """
        Display the current state of all virtual actuators.
        """

        print("\n[4] ACTUATOR RESPONSE")
        print("-" * 45)

        print(f"Cooling         : {self.state.cooling}")
        print(f"Ventilation     : {self.state.ventilation}")
        print(f"Lighting        : {self.state.lighting}")


if __name__ == "__main__":

    from decision.decision_engine import CognitiveDecisionEngine

    engine = CognitiveDecisionEngine()

    test_prediction = {
        "occupancy_class": 3,
        "occupancy_state": "High Occupancy",
        "confidence": 0.95,
    }

    decision = engine.make_decision(
        prediction=test_prediction,
        temperature=29.0,
        co2=1100.0,
        light=50.0,
        sound=0.65,
    )

    actuator_system = VirtualActuatorSystem()

    actuator_system.apply_decision(decision)

    print("\nVIRTUAL ACTUATOR TEST")
    print("=" * 50)

    actuator_system.display_state()