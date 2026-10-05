"""
Cognitive IoT - Temporal Shift Impact Analysis

Measures whether the temporal distribution shift detected
between training and testing periods affects model
prediction performance and confidence.

This module does NOT modify the trained model.
It performs an independent evaluation experiment.
"""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_occupancy_features.csv"
)


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

TARGET_COLUMN = "Room_Occupancy_Count"

TRAIN_RATIO = 0.80

SENSOR_COLUMNS = {
    "Temperature": "temperature_mean",
    "Lighting": "light_mean",
    "Sound": "sound_mean",
    "CO2": "co2",
    "Motion": "pir_activity",
}


# -------------------------------------------------------------------
# CREATE CONTEXT FEATURES
# -------------------------------------------------------------------

def create_context_features(dataframe, train_dataframe):
    """
    Create normalized context features.

    The scaler for each sensor is fitted ONLY on the
    training period.

    This prevents information from the testing period
    leaking into the training process.
    """

    contexts = {}

    sensor_mapping = {
        "temperature": "temperature_mean",
        "lighting": "light_mean",
        "sound": "sound_mean",
        "co2": "co2",
        "motion": "pir_activity",
    }

    for name, column in sensor_mapping.items():

        scaler = MinMaxScaler()

        scaler.fit(
            train_dataframe[column]
            .to_numpy()
            .reshape(-1, 1)
        )

        contexts[f"{name}_context"] = (
            scaler.transform(
                dataframe[column]
                .to_numpy()
                .reshape(-1, 1)
            )
            .ravel()
        )

    result = pd.DataFrame(
        {
            "temperature_context":
                contexts["temperature_context"],

            "lighting_context":
                contexts["lighting_context"],

            "sound_context":
                contexts["sound_context"],

            "co2_context":
                contexts["co2_context"],

            "motion_context":
                contexts["motion_context"],
        },
        index=dataframe.index,
    )

    result["unified_context_score"] = result.mean(axis=1)

    return result


# -------------------------------------------------------------------
# SHIFT SEVERITY
# -------------------------------------------------------------------

def classify_shift(change):

    absolute_change = abs(change)

    if absolute_change < 5:
        return "MINIMAL"

    elif absolute_change < 15:
        return "MODERATE"

    elif absolute_change < 30:
        return "SIGNIFICANT"

    elif absolute_change < 50:
        return "VERY LARGE"

    else:
        return "EXTREME"


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print("       TEMPORAL SHIFT IMPACT ANALYSIS")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    dataframe = pd.read_csv(DATA_FILE)

    print(
        f"Dataset rows: {len(dataframe)}"
    )

    print()

    # ---------------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # ---------------------------------------------------------------

    if "timestamp" not in dataframe.columns:

        raise ValueError(
            "Timestamp column not found."
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    dataframe = (
        dataframe
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # TEMPORAL SPLIT
    # ---------------------------------------------------------------

    split_index = int(
        len(dataframe) * TRAIN_RATIO
    )

    train_dataframe = dataframe.iloc[
        :split_index
    ].copy()

    test_dataframe = dataframe.iloc[
        split_index:
    ].copy()

    print(
        f"Training samples: "
        f"{len(train_dataframe)}"
    )

    print(
        f"Testing samples:  "
        f"{len(test_dataframe)}"
    )

    print()

    print("Training period:")

    print(
        f"{train_dataframe['timestamp'].min()} "
        f"-> "
        f"{train_dataframe['timestamp'].max()}"
    )

    print()

    print("Testing period:")

    print(
        f"{test_dataframe['timestamp'].min()} "
        f"-> "
        f"{test_dataframe['timestamp'].max()}"
    )

    print()

    # ---------------------------------------------------------------
    # CREATE FEATURES
    # ---------------------------------------------------------------

    train_context = create_context_features(
        train_dataframe,
        train_dataframe,
    )

    test_context = create_context_features(
        test_dataframe,
        train_dataframe,
    )

    # ---------------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------------

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    # ---------------------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------------------

    print("=" * 70)
    print("TRAINING TEMPORAL MODEL")
    print("=" * 70)
    print()

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        train_context,
        y_train,
    )

    print("Model training completed.")

    print()

    # ---------------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------------

    predictions = model.predict(
        test_context
    )

    probabilities = model.predict_proba(
        test_context
    )

    # Maximum class probability for every prediction
    confidence = probabilities.max(
        axis=1
    )

    # ---------------------------------------------------------------
    # ACCURACY
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print()

    print(
        f"Temporal Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    # ---------------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------------

    average_confidence = confidence.mean()

    low_confidence_threshold = 0.60

    low_confidence_count = (
        confidence < low_confidence_threshold
    ).sum()

    low_confidence_percentage = (
        low_confidence_count
        / len(confidence)
        * 100
    )

    print(
        f"Average prediction confidence: "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"Predictions below "
        f"{low_confidence_threshold * 100:.0f}% confidence: "
        f"{low_confidence_percentage:.2f}%"
    )

    print()

    # ---------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------------

    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------------

    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()

    # ---------------------------------------------------------------
    # SENSOR SHIFT ANALYSIS
    # ---------------------------------------------------------------

    print("=" * 70)
    print("SENSOR DISTRIBUTION SHIFT")
    print("=" * 70)
    print()

    shift_results = []

    for sensor_name, column in SENSOR_COLUMNS.items():

        train_mean = (
            train_dataframe[column]
            .mean()
        )

        test_mean = (
            test_dataframe[column]
            .mean()
        )

        if train_mean != 0:

            change = (
                (test_mean - train_mean)
                / train_mean
                * 100
            )

        else:

            change = 0

        severity = classify_shift(
            change
        )

        shift_results.append(
            {
                "Sensor": sensor_name,
                "Train Mean": train_mean,
                "Test Mean": test_mean,
                "Change %": change,
                "Severity": severity,
            }
        )

    shift_dataframe = pd.DataFrame(
        shift_results
    )

    print(
        shift_dataframe.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()

    # ---------------------------------------------------------------
    # FIND STRONGEST SHIFT
    # ---------------------------------------------------------------

    strongest_shift = max(
        shift_results,
        key=lambda row: abs(row["Change %"])
    )

    # ---------------------------------------------------------------
    # COGNITIVE ASSESSMENT
    # ---------------------------------------------------------------

    print("=" * 70)
    print("COGNITIVE ASSESSMENT")
    print("=" * 70)
    print()

    print(
        f"Strongest distribution shift: "
        f"{strongest_shift['Sensor']}"
    )

    print(
        f"Shift magnitude: "
        f"{abs(strongest_shift['Change %']):.2f}%"
    )

    print(
        f"Shift severity: "
        f"{strongest_shift['Severity']}"
    )

    print()

    if strongest_shift["Severity"] == "EXTREME":

        print(
            "Assessment: EXTREME ENVIRONMENTAL SHIFT"
        )

        print(
            "Recommendation: Increase monitoring "
            "and consider model adaptation."
        )

    elif strongest_shift["Severity"] == "VERY LARGE":

        print(
            "Assessment: VERY LARGE ENVIRONMENTAL SHIFT"
        )

        print(
            "Recommendation: Monitor prediction "
            "reliability closely."
        )

    elif strongest_shift["Severity"] == "SIGNIFICANT":

        print(
            "Assessment: SIGNIFICANT ENVIRONMENTAL SHIFT"
        )

        print(
            "Recommendation: Continue monitoring "
            "for performance degradation."
        )

    elif strongest_shift["Severity"] == "MODERATE":

        print(
            "Assessment: MODERATE ENVIRONMENTAL SHIFT"
        )

        print(
            "Recommendation: Continue normal operation "
            "with monitoring."
        )

    else:

        print(
            "Assessment: ENVIRONMENT IS STABLE"
        )

        print(
            "Recommendation: Continue normal operation."
        )

    print()

    # ---------------------------------------------------------------
    # CONFIDENCE ASSESSMENT
    # ---------------------------------------------------------------

    print("MODEL RELIABILITY")
    print("-" * 70)

    if average_confidence >= 0.80:

        print(
            "Prediction confidence: HIGH"
        )

    elif average_confidence >= 0.60:

        print(
            "Prediction confidence: MODERATE"
        )

    else:

        print(
            "Prediction confidence: LOW"
        )

    print(
        f"Average confidence: "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"Low-confidence predictions: "
        f"{low_confidence_percentage:.2f}%"
    )

    print()

    print("=" * 70)
    print(
        "Temporal shift impact analysis completed."
    )
    print("=" * 70)


# -------------------------------------------------------------------
# PROGRAM ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()