"""
Cognitive IoT - Leakage-Aware Model Evaluation

Evaluates the occupancy model while ensuring that all
normalization parameters are learned only from the training set.

The existing model is not modified.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_FILE = Path(
    "data/processed/room_occupancy_features.csv"
)

MODEL_FILE = Path(
    "ai/model/occupancy_model_leakage_aware.pkl"
)


FEATURE_COLUMNS = [
    "temperature_mean",
    "temperature_std",
    "temperature_range",

    "light_mean",
    "light_std",
    "light_range",

    "sound_mean",
    "sound_std",
    "sound_range",

    "co2",
    "co2_slope",

    "pir_activity",
    "pir_sensor_count",
]


TARGET_COLUMN = (
    "Room_Occupancy_Count"
)


def load_dataset():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    dataframe = pd.read_csv(
        DATA_FILE
    )

    return dataframe


def create_context_features(
    dataframe,
    scalers=None,
    fit_scalers=False,
):
    """
    Create leakage-aware context features.

    Each context is normalized using a scaler.

    During training:
        scaler is fitted on training data.

    During testing:
        training scaler is reused.
    """

    dataframe = dataframe.copy()

    context_definitions = {

        "temperature_context": [
            "temperature_mean"
        ],

        "lighting_context": [
            "light_mean"
        ],

        "sound_context": [
            "sound_mean"
        ],

        "co2_context": [
            "co2"
        ],

        "motion_context": [
            "pir_activity"
        ],
    }

    if scalers is None:

        scalers = {}

    context_data = {}

    for context_name, columns in (
        context_definitions.items()
    ):

        values = dataframe[
            columns
        ].mean(axis=1).values.reshape(
            -1,
            1
        )

        if fit_scalers:

            scaler = MinMaxScaler()

            normalized_values = (
                scaler.fit_transform(
                    values
                )
            )

            scalers[
                context_name
            ] = scaler

        else:

            if context_name not in scalers:

                raise ValueError(
                    f"No training scaler found "
                    f"for {context_name}"
                )

            scaler = scalers[
                context_name
            ]

            normalized_values = (
                scaler.transform(
                    values
                )
            )

        context_data[
            context_name
        ] = normalized_values.flatten()

    result = pd.DataFrame(
        context_data,
        index=dataframe.index
    )

    result[
        "unified_context_score"
    ] = result.mean(
        axis=1
    )

    return result, scalers


def main():

    print()

    print(
        "========================================"
    )

    print(
        "   LEAKAGE-AWARE MODEL EVALUATION"
    )

    print(
        "========================================"
    )

    print()

    dataframe = load_dataset()

    print(
        "Dataset rows:",
        len(dataframe)
    )

    print()

    # --------------------------------------------
    # Validate required columns
    # --------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        print(
            "Missing columns:"
        )

        for column in missing_columns:

            print(
                " -",
                column
            )

        raise ValueError(
            "The feature-engineered dataset "
            "does not contain the required columns."
        )

    # --------------------------------------------
    # Raw feature matrix and target
    # --------------------------------------------

    X_raw = dataframe[
        FEATURE_COLUMNS
    ]

    y = dataframe[
        TARGET_COLUMN
    ]

    # --------------------------------------------
    # STEP 1
    # Split before normalization
    # --------------------------------------------

    (
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
    ) = train_test_split(

        X_raw,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y,
    )

    print(
        "Training samples:",
        len(X_train_raw)
    )

    print(
        "Test samples:",
        len(X_test_raw)
    )

    print()

    # --------------------------------------------
    # STEP 2
    # Leakage-aware sensor fusion
    # --------------------------------------------

    (
        train_context,
        scalers,
    ) = create_context_features(

        X_train_raw,

        fit_scalers=True,
    )

    (
        test_context,
        _,
    ) = create_context_features(

        X_test_raw,

        scalers=scalers,

        fit_scalers=False,
    )

    print(
        "Context features created."
    )

    print()

    # --------------------------------------------
    # STEP 3
    # Train fresh model
    # --------------------------------------------

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1,
    )

    model.fit(
        train_context,
        y_train
    )

    # --------------------------------------------
    # STEP 4
    # Evaluate unseen test data
    # --------------------------------------------

    predictions = model.predict(
        test_context
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "Leakage-aware accuracy:",
        f"{accuracy:.4%}"
    )

    print()

    print(
        "Classification Report"
    )

    print(
        "----------------------"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix"
    )

    print(
        "----------------"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()

    # --------------------------------------------
    # STEP 5
    # Feature importance
    # --------------------------------------------

    print(
        "Feature Importance"
    )

    print(
        "------------------"
    )

    importance = pd.Series(

        model.feature_importances_,

        index=train_context.columns,

    ).sort_values(
        ascending=False
    )

    for feature, value in (
        importance.items()
    ):

        print(
            f"{feature:<25}",
            f"{value:.4f}"
        )

    # --------------------------------------------
    # STEP 6
    # Save separate model
    # --------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(

        {
            "model": model,

            "scalers": scalers,

            "features":
                list(
                    train_context.columns
                ),
        },

        MODEL_FILE,
    )

    print()

    print(
        "Leakage-aware model saved:"
    )

    print(
        MODEL_FILE
    )

    print()

    print(
        "Evaluation completed."
    )


if __name__ == "__main__":

    main()