"""
Cognitive IoT - Sensor Fusion

Combines engineered features from multiple sensor types
into a unified representation for the AI model.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_occupancy_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fused_sensor_data.csv"
)


# -------------------------------------------------------------------
# FEATURE GROUPS
# -------------------------------------------------------------------

TEMPERATURE_FEATURES = [
    "temperature_mean",
    "temperature_std",
    "temperature_range",
]

LIGHT_FEATURES = [
    "light_mean",
    "light_std",
    "light_range",
]

SOUND_FEATURES = [
    "sound_mean",
    "sound_std",
    "sound_range",
]

CO2_FEATURES = [
    "co2",
    "co2_slope",
]

PIR_FEATURES = [
    "pir_activity",
    "pir_sensor_count",
]


# -------------------------------------------------------------------
# MAIN SENSOR FUSION FUNCTION
# -------------------------------------------------------------------

def perform_sensor_fusion():
    """Create a unified sensor representation."""

    print("=" * 70)
    print("COGNITIVE IoT - SENSOR FUSION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load engineered features
    # ---------------------------------------------------------------

    print("\nLoading engineered feature dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # ---------------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------------

    required_features = (
        TEMPERATURE_FEATURES
        + LIGHT_FEATURES
        + SOUND_FEATURES
        + CO2_FEATURES
        + PIR_FEATURES
    )

    required_columns = (
        ["timestamp"]
        + required_features
        + ["Room_Occupancy_Count"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------------
    # Create sensor-group scores
    # ---------------------------------------------------------------

    print("\nCreating sensor group representations...")

    # Temperature context
    df["temperature_context"] = df[
        TEMPERATURE_FEATURES
    ].mean(axis=1)

    # Lighting context
    df["lighting_context"] = df[
        LIGHT_FEATURES
    ].mean(axis=1)

    # Sound context
    df["sound_context"] = df[
        SOUND_FEATURES
    ].mean(axis=1)

    # CO2 context
    df["co2_context"] = df[
        CO2_FEATURES
    ].mean(axis=1)

    # PIR / movement context
    df["motion_context"] = df[
        PIR_FEATURES
    ].mean(axis=1)

    # ---------------------------------------------------------------
    # Create unified sensor context
    # ---------------------------------------------------------------

    print("Creating unified sensor context...")

    context_features = [
        "temperature_context",
        "lighting_context",
        "sound_context",
        "co2_context",
        "motion_context",
    ]

    # Normalized sensor contribution
    normalized_context = pd.DataFrame(index=df.index)

    for feature in context_features:
        minimum = df[feature].min()
        maximum = df[feature].max()

        if maximum == minimum:
            normalized_context[feature] = 0.0
        else:
            normalized_context[feature] = (
                (df[feature] - minimum)
                / (maximum - minimum)
            )

    # Combined context score
    df["unified_context_score"] = normalized_context.mean(axis=1)

    # ---------------------------------------------------------------
    # Select final fused representation
    # ---------------------------------------------------------------

    fused_columns = [
        "timestamp",
        "temperature_context",
        "lighting_context",
        "sound_context",
        "co2_context",
        "motion_context",
        "unified_context_score",
        "Room_Occupancy_Count",
    ]

    fused_df = df[fused_columns].copy()

    # ---------------------------------------------------------------
    # Save fused dataset
    # ---------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fused_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------

    print(
        f"\nFinal fused dataset shape: "
        f"{fused_df.shape[0]} rows × "
        f"{fused_df.shape[1]} columns"
    )

    print("\nFused features:")

    for column in fused_columns:
        print(f"  - {column}")

    print(
        f"\nFused sensor data saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nSensor fusion complete.")


# -------------------------------------------------------------------
# PROGRAM ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    perform_sensor_fusion()