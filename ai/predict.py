"""
Cognitive IoT - Occupancy Predictor

Loads the trained occupancy model and the exact scaler used
during training.

Pipeline:

Raw sensor data
    ↓
Training scaler
    ↓
Context features
    ↓
Random Forest model
    ↓
Occupancy prediction
"""

from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "ai"
    / "model"
    / "occupancy_model.pkl"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "ai"
    / "model"
    / "occupancy_scaler.pkl"
)


# ============================================================
# FEATURES
# ============================================================

RAW_FEATURES = [
    "temperature_mean",
    "light_mean",
    "sound_mean",
    "co2",
    "pir_activity",
]

CONTEXT_FEATURES = [
    "temperature_context",
    "lighting_context",
    "sound_context",
    "co2_context",
    "motion_context",
    "unified_context_score",
]


# ============================================================
# OCCUPANCY LABELS
# ============================================================

OCCUPANCY_LABELS = {
    0: "Empty",
    1: "Low Occupancy",
    2: "Medium Occupancy",
    3: "High Occupancy",
}


# ============================================================
# PREDICTOR
# ============================================================

class OccupancyPredictor:

    def __init__(
        self,
        model_file=MODEL_FILE,
        scaler_file=SCALER_FILE,
    ):

        self.model_file = Path(model_file)
        self.scaler_file = Path(scaler_file)

        if not self.model_file.exists():
            raise FileNotFoundError(
                f"Model file not found:\n"
                f"{self.model_file}"
            )

        if not self.scaler_file.exists():
            raise FileNotFoundError(
                f"Scaler file not found:\n"
                f"{self.scaler_file}"
            )

        self.model = joblib.load(
            self.model_file
        )

        self.scaler = joblib.load(
            self.scaler_file
        )

    # --------------------------------------------------------
    # Context feature creation
    # --------------------------------------------------------

    def create_context_features(
        self,
        sensor_data,
    ):
        """
        Convert raw sensor readings into the same
        context representation used during training.
        """

        raw_data = pd.DataFrame(
            [
                {
                    "temperature_mean":
                        sensor_data["temperature"],

                    "light_mean":
                        sensor_data["light"],

                    "sound_mean":
                        sensor_data["sound"],

                    "co2":
                        sensor_data["co2"],

                    "pir_activity":
                        sensor_data["motion"],
                }
            ]
        )

        scaled_values = self.scaler.transform(
            raw_data[RAW_FEATURES]
        )

        scaled = pd.DataFrame(
            scaled_values,
            columns=RAW_FEATURES,
        )

        context = pd.DataFrame()

        context["temperature_context"] = (
            scaled["temperature_mean"]
        )

        context["lighting_context"] = (
            scaled["light_mean"]
        )

        context["sound_context"] = (
            scaled["sound_mean"]
        )

        context["co2_context"] = (
            scaled["co2"]
        )

        context["motion_context"] = (
            scaled["pir_activity"]
        )

        context["unified_context_score"] = (
            context[
                [
                    "temperature_context",
                    "lighting_context",
                    "sound_context",
                    "co2_context",
                    "motion_context",
                ]
            ].mean(axis=1)
        )

        return context[
            CONTEXT_FEATURES
        ]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(
        self,
        sensor_data,
    ):
        """
        Predict occupancy from raw sensor readings.

        Expected input:

        {
            "temperature": float,
            "light": float,
            "sound": float,
            "co2": float,
            "motion": float
        }
        """

        context_features = (
            self.create_context_features(
                sensor_data
            )
        )

        occupancy_class = int(
            self.model.predict(
                context_features
            )[0]
        )

        probabilities = (
            self.model.predict_proba(
                context_features
            )[0]
        )

        confidence = float(
            probabilities[occupancy_class]
        )

        occupancy_state = OCCUPANCY_LABELS.get(
            occupancy_class,
            "Unknown",
        )

        return {
            "occupancy_class": occupancy_class,
            "occupancy_state": occupancy_state,
            "confidence": confidence,
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("       COGNITIVE IoT OCCUPANCY PREDICTOR")
    print("=" * 50)

    predictor = OccupancyPredictor()

    test_sensor_data = {
        "temperature": 28.5,
        "light": 120.0,
        "sound": 0.35,
        "co2": 850.0,
        "motion": 1.0,
    }

    print("\nTest sensor data:")

    for key, value in test_sensor_data.items():
        print(
            f"  {key:<12}: {value}"
        )

    prediction = predictor.predict(
        test_sensor_data
    )

    print("\nPrediction:")
    print(
        f"  Occupancy Class : "
        f"{prediction['occupancy_class']}"
    )

    print(
        f"  Occupancy State : "
        f"{prediction['occupancy_state']}"
    )

    print(
        f"  Confidence      : "
        f"{prediction['confidence'] * 100:.2f}%"
    )

    print("\nPrediction completed successfully.")