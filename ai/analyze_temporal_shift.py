"""
Cognitive IoT - Temporal Distribution Shift Analysis

Compares the training and temporal test periods to identify
changes in sensor distributions and occupancy patterns.

This helps explain why temporal model performance may differ
from random-split performance.
"""

from pathlib import Path

import pandas as pd


DATA_FILE = Path(
    "data/processed/room_occupancy_features.csv"
)

TARGET_COLUMN = "Room_Occupancy_Count"

TRAIN_RATIO = 0.80


NUMERIC_COLUMNS = [
    "temperature_mean",
    "light_mean",
    "sound_mean",
    "co2",
    "pir_activity",
]


DISPLAY_NAMES = {
    "temperature_mean": "Temperature",
    "light_mean": "Lighting",
    "sound_mean": "Sound",
    "co2": "CO2",
    "pir_activity": "Motion",
}


def print_separator():
    print("-" * 60)


def main():

    print("=" * 60)
    print("       TEMPORAL DISTRIBUTION SHIFT ANALYSIS")
    print("=" * 60)
    print()

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    dataframe = pd.read_csv(DATA_FILE)

    print(f"Dataset rows: {len(dataframe)}")
    print()

    # --------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------

    if "timestamp" not in dataframe.columns:
        raise ValueError(
            "Timestamp column not found."
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    dataframe = dataframe.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Temporal split
    # --------------------------------------------------

    split_index = int(
        len(dataframe) * TRAIN_RATIO
    )

    train = dataframe.iloc[
        :split_index
    ].copy()

    test = dataframe.iloc[
        split_index:
    ].copy()

    print("TRAINING PERIOD")
    print(
        f"{train['timestamp'].min()} "
        f"→ "
        f"{train['timestamp'].max()}"
    )

    print()

    print("TESTING PERIOD")
    print(
        f"{test['timestamp'].min()} "
        f"→ "
        f"{test['timestamp'].max()}"
    )

    print()

    # --------------------------------------------------
    # Sensor distribution comparison
    # --------------------------------------------------

    print("=" * 60)
    print("SENSOR DISTRIBUTION COMPARISON")
    print("=" * 60)
    print()

    rows = []

    for column in NUMERIC_COLUMNS:

        train_mean = train[column].mean()
        test_mean = test[column].mean()

        train_std = train[column].std()
        test_std = test[column].std()

        train_min = train[column].min()
        test_min = test[column].min()

        train_max = train[column].max()
        test_max = test[column].max()

        mean_change = (
            ((test_mean - train_mean) / train_mean) * 100
            if train_mean != 0
            else 0
        )

        rows.append(
            {
                "Sensor": DISPLAY_NAMES[column],
                "Train Mean": train_mean,
                "Test Mean": test_mean,
                "Mean Change %": mean_change,
                "Train Std": train_std,
                "Test Std": test_std,
                "Train Min": train_min,
                "Test Min": test_min,
                "Train Max": train_max,
                "Test Max": test_max,
            }
        )

    comparison = pd.DataFrame(rows)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print()

    # --------------------------------------------------
    # Occupancy distribution
    # --------------------------------------------------

    print("=" * 60)
    print("OCCUPANCY DISTRIBUTION")
    print("=" * 60)
    print()

    train_counts = (
        train[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    test_counts = (
        test[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    labels = {
        0: "Empty",
        1: "Low Occupancy",
        2: "Medium Occupancy",
        3: "High Occupancy",
    }

    occupancy_rows = []

    for class_id in sorted(
        set(train_counts.index)
        | set(test_counts.index)
    ):

        train_count = train_counts.get(
            class_id,
            0
        )

        test_count = test_counts.get(
            class_id,
            0
        )

        train_percentage = (
            train_count / len(train)
        ) * 100

        test_percentage = (
            test_count / len(test)
        ) * 100

        occupancy_rows.append(
            {
                "Class": labels.get(
                    class_id,
                    str(class_id)
                ),
                "Train Count": train_count,
                "Train %": train_percentage,
                "Test Count": test_count,
                "Test %": test_percentage,
            }
        )

    occupancy_comparison = pd.DataFrame(
        occupancy_rows
    )

    print(
        occupancy_comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()

    # --------------------------------------------------
    # Context shift summary
    # --------------------------------------------------

    print("=" * 60)
    print("SHIFT SUMMARY")
    print("=" * 60)
    print()

    print(
        "The following sensor changes are based on the "
        "difference between the training and testing periods."
    )

    print()

    for _, row in comparison.iterrows():

        change = row["Mean Change %"]

        direction = (
            "increased"
            if change > 0
            else "decreased"
            if change < 0
            else "remained stable"
        )

        print(
            f"{row['Sensor']:<15} "
            f"{direction:<12} "
            f"{abs(change):.2f}%"
        )

    print()

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()

    print(
        "If sensor distributions or occupancy patterns "
        "change substantially between the training and "
        "testing periods, the temporal accuracy drop may "
        "be caused by distribution shift."
    )

    print()

    print(
        "This analysis does not modify the model. "
        "It only investigates differences between the "
        "two time periods."
    )

    print()

    print(
        "Temporal shift analysis completed."
    )


if __name__ == "__main__":
    main()