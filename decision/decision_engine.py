"""
Cognitive IoT - Cognitive Decision Engine

Converts AI understanding and environmental sensor data
into autonomous actuator decisions.
"""

from dataclasses import dataclass


@dataclass
class DecisionResult:
    occupancy_state: str
    occupancy_confidence: float

    cooling: str
    ventilation: str
    lighting: str

    priority: str
    reasons: list


class CognitiveDecisionEngine:

    def __init__(self):

        # Environmental thresholds
        self.temperature_high = 27.0
        self.temperature_low = 20.0

        self.co2_high = 800.0
        self.co2_critical = 1200.0

        self.light_low = 100.0
        self.light_high = 700.0

        self.sound_high = 0.50

    def make_decision(
        self,
        prediction,
        temperature,
        co2,
        light,
        sound,
    ):

        occupancy_class = prediction["occupancy_class"]
        occupancy_state = prediction["occupancy_state"]
        confidence = prediction["confidence"]

        reasons = []

        # --------------------------------------------------
        # COOLING DECISION
        # --------------------------------------------------

        if occupancy_class == 0:

            cooling = "OFF"

            reasons.append(
                "Room is empty, so cooling is disabled."
            )

        elif temperature >= self.temperature_high:

            cooling = "HIGH"

            reasons.append(
                "Temperature is above the comfort threshold."
            )

        elif temperature >= 24.0:

            cooling = "MEDIUM"

            reasons.append(
                "Temperature is moderately high."
            )

        else:

            cooling = "LOW"

            reasons.append(
                "Temperature is within a comfortable range."
            )

        # --------------------------------------------------
        # VENTILATION DECISION
        # --------------------------------------------------

        if occupancy_class == 0:

            ventilation = "LOW"

            reasons.append(
                "Room is empty, so ventilation is minimized."
            )

        elif co2 >= self.co2_critical:

            ventilation = "HIGH"

            reasons.append(
                "CO2 level is critically high."
            )

        elif co2 >= self.co2_high:

            ventilation = "MEDIUM"

            reasons.append(
                "CO2 level is above the ventilation threshold."
            )

        else:

            ventilation = "LOW"

            reasons.append(
                "CO2 level is within the acceptable range."
            )

        # --------------------------------------------------
        # LIGHTING DECISION
        # --------------------------------------------------

        if occupancy_class == 0:

            lighting = "OFF"

            reasons.append(
                "Room is empty, so lighting is switched off."
            )

        elif light < self.light_low:

            lighting = "HIGH"

            reasons.append(
                "Ambient light is too low."
            )

        elif light > self.light_high:

            lighting = "LOW"

            reasons.append(
                "Ambient light is already high."
            )

        else:

            lighting = "MEDIUM"

            reasons.append(
                "Ambient light is at a moderate level."
            )

        # --------------------------------------------------
        # PRIORITY
        # --------------------------------------------------

        if co2 >= self.co2_critical or temperature >= 30:

            priority = "CRITICAL"

        elif (
            occupancy_class >= 2
            or co2 >= self.co2_high
            or temperature >= self.temperature_high
            or sound >= self.sound_high
        ):

            priority = "HIGH"

        else:

            priority = "NORMAL"

        return DecisionResult(
            occupancy_state=occupancy_state,
            occupancy_confidence=confidence,
            cooling=cooling,
            ventilation=ventilation,
            lighting=lighting,
            priority=priority,
            reasons=reasons,
        )


if __name__ == "__main__":

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

    print("\nCOGNITIVE DECISION")
    print("=" * 50)

    print(
        f"Occupancy       : "
        f"{decision.occupancy_state}"
    )

    print(
        f"Confidence      : "
        f"{decision.occupancy_confidence * 100:.2f}%"
    )

    print(
        f"Cooling         : "
        f"{decision.cooling}"
    )

    print(
        f"Ventilation     : "
        f"{decision.ventilation}"
    )

    print(
        f"Lighting        : "
        f"{decision.lighting}"
    )

    print(
        f"Priority        : "
        f"{decision.priority}"
    )

    print("\nDecision Reasons")
    print("-" * 50)

    for reason in decision.reasons:
        print(f"- {reason}")