"""
Cognitive IoT - Temporal Model Evaluation

Trains the model on earlier observations and evaluates it
on later unseen observations.

This provides a stronger time-aware validation than a
random train/test split.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler


DATA_FILE = Path("data/processed/room_occupancy_features.csv")
MODEL_FILE = Path("ai/model/occupancy_model_temporal.pkl")

TARGET_COLUMN = "Room_Occupancy_Count"

TRAIN_RATIO = 0.80


def create_context_features(dataframe, train_dataframe):
    """
    Create context features using normalization parameters
    learned only from the training data.
    """

    train_values = {
        "temperature": train_dataframe["temperature_mean"],
        "lighting": train_dataframe["light_mean"],
        "sound": train_dataframe["sound_mean"],
        "co2": train_dataframe["co2"],
        "motion": train_dataframe["pir_activity"],
    }

    test_values = {
        "temperature": dataframe["temperature_mean"],
        "lighting": dataframe["light_mean"],
        "sound": dataframe["sound_mean"],
        "co2": dataframe["co2"],
        "motion": dataframe["pir_activity"],
    }

    contexts = {}

    for name in train_values:

        scaler = MinMaxScaler()

        scaler.fit(
            train_values[name].to_numpy().reshape(-1, 1)
        )

        contexts[f"{name}_context"] = scaler.transform(
            test_values[name].to_numpy().reshape(-1, 1)
        ).ravel()

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


def main():

    print("=" * 40)
    print("   TEMPORAL MODEL EVALUATION")
    print("=" * 40)
    print()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    dataframe = pd.read_csv(DATA_FILE)

    print(f"Dataset rows: {len(dataframe)}")
    print()

    # --------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------

    if "timestamp" in dataframe.columns:

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"]
        )

        dataframe = dataframe.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    else:

        raise ValueError(
            "Timestamp column not found in dataset."
        )

    # --------------------------------------------------
    # Temporal train/test split
    # --------------------------------------------------

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
        f"Training samples: {len(train_dataframe)}"
    )

    print(
        f"Test samples: {len(test_dataframe)}"
    )

    print()

    print(
        "Training period:"
    )

    print(
        f"{train_dataframe['timestamp'].min()} "
        f"→ "
        f"{train_dataframe['timestamp'].max()}"
    )

    print()

    print(
        "Testing period:"
    )

    print(
        f"{test_dataframe['timestamp'].min()} "
        f"→ "
        f"{test_dataframe['timestamp'].max()}"
    )

    print()

    # --------------------------------------------------
    # Create context features
    # --------------------------------------------------

    train_context = create_context_features(
        train_dataframe,
        train_dataframe,
    )

    test_context = create_context_features(
        test_dataframe,
        train_dataframe,
    )

    print("Context features created.")
    print()

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    predictions = model.predict(
        test_context
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        f"Temporal accuracy: {accuracy * 100:.4f}%"
    )

    print()

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("Classification Report")
    print("----------------------")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    print("Confusion Matrix")
    print("----------------")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    feature_importance = pd.Series(
        model.feature_importances_,
        index=train_context.columns,
    ).sort_values(
        ascending=False
    )

    print("Feature Importance")
    print("------------------")

    for feature, importance in feature_importance.items():

        print(
            f"{feature:<25}"
            f"{importance:.4f}"
        )

    print()

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "Temporal model saved:"
    )

    print(
        MODEL_FILE
    )

    print()

    print(
        "Temporal evaluation completed."
    )


if __name__ == "__main__":
    main()