"""
Feature engineering module for the Cognitive IoT project.

This module converts multiple raw sensor readings into
meaningful environmental features that can be used by the
sensor-fusion and AI components.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_occupancy_cleaned.csv"
)

OUTPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_occupancy_features.csv"
)


# -------------------------------------------------------------------
# Sensor groups
# -------------------------------------------------------------------

TEMPERATURE_SENSORS = [
    "S1_Temp",
    "S2_Temp",
    "S3_Temp",
    "S4_Temp",
]

LIGHT_SENSORS = [
    "S1_Light",
    "S2_Light",
    "S3_Light",
    "S4_Light",
]

SOUND_SENSORS = [
    "S1_Sound",
    "S2_Sound",
    "S3_Sound",
    "S4_Sound",
]

PIR_SENSORS = [
    "S6_PIR",
    "S7_PIR",
]


# -------------------------------------------------------------------
# Load cleaned dataset
# -------------------------------------------------------------------

def load_cleaned_dataset():
    """
    Load the cleaned Room Occupancy Estimation dataset.

    Returns:
        pandas.DataFrame: Cleaned dataset.
    """

    if not INPUT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {INPUT_DATA_PATH}\n"
            "Run preprocessing.data_cleaning first."
        )

    return pd.read_csv(INPUT_DATA_PATH)


# -------------------------------------------------------------------
# Temperature features
# -------------------------------------------------------------------

def create_temperature_features(dataframe):
    """
    Create fused temperature features from four temperature sensors.

    Features:
        temperature_mean
        temperature_std
        temperature_range
    """

    dataframe = dataframe.copy()

    dataframe["temperature_mean"] = dataframe[
        TEMPERATURE_SENSORS
    ].mean(axis=1)

    dataframe["temperature_std"] = dataframe[
        TEMPERATURE_SENSORS
    ].std(axis=1).fillna(0)

    dataframe["temperature_range"] = (
        dataframe[TEMPERATURE_SENSORS].max(axis=1)
        - dataframe[TEMPERATURE_SENSORS].min(axis=1)
    )

    return dataframe


# -------------------------------------------------------------------
# Light features
# -------------------------------------------------------------------

def create_light_features(dataframe):
    """
    Create fused lighting features from four light sensors.

    Features:
        light_mean
        light_std
        light_range
    """

    dataframe = dataframe.copy()

    dataframe["light_mean"] = dataframe[
        LIGHT_SENSORS
    ].mean(axis=1)

    dataframe["light_std"] = dataframe[
        LIGHT_SENSORS
    ].std(axis=1).fillna(0)

    dataframe["light_range"] = (
        dataframe[LIGHT_SENSORS].max(axis=1)
        - dataframe[LIGHT_SENSORS].min(axis=1)
    )

    return dataframe


# -------------------------------------------------------------------
# Sound features
# -------------------------------------------------------------------

def create_sound_features(dataframe):
    """
    Create fused sound features from four sound sensors.

    Features:
        sound_mean
        sound_std
        sound_range
    """

    dataframe = dataframe.copy()

    dataframe["sound_mean"] = dataframe[
        SOUND_SENSORS
    ].mean(axis=1)

    dataframe["sound_std"] = dataframe[
        SOUND_SENSORS
    ].std(axis=1).fillna(0)

    dataframe["sound_range"] = (
        dataframe[SOUND_SENSORS].max(axis=1)
        - dataframe[SOUND_SENSORS].min(axis=1)
    )

    return dataframe


# -------------------------------------------------------------------
# PIR features
# -------------------------------------------------------------------

def create_pir_features(dataframe):
    """
    Combine the two PIR sensors into a single motion/activity feature.

    Features:
        pir_activity
        pir_sensor_count
    """

    dataframe = dataframe.copy()

    dataframe["pir_activity"] = (
        dataframe[PIR_SENSORS].max(axis=1)
    )

    dataframe["pir_sensor_count"] = (
        dataframe[PIR_SENSORS].sum(axis=1)
    )

    return dataframe


# -------------------------------------------------------------------
# CO2 features
# -------------------------------------------------------------------

def create_co2_features(dataframe):
    """
    Prepare CO2-related features.

    The original dataset already provides:
        S5_CO2
        S5_CO2_Slope

    These are renamed to more meaningful project-level names.
    """

    dataframe = dataframe.copy()

    dataframe["co2"] = dataframe["S5_CO2"]

    dataframe["co2_slope"] = dataframe["S5_CO2_Slope"]

    return dataframe


# -------------------------------------------------------------------
# Build fused feature dataset
# -------------------------------------------------------------------

def create_fused_features(dataframe):
    """
    Create all fused environmental features.

    Returns:
        pandas.DataFrame: Dataset containing engineered features
        and the original occupancy target.
    """

    dataframe = dataframe.copy()

    dataframe = create_temperature_features(dataframe)
    dataframe = create_light_features(dataframe)
    dataframe = create_sound_features(dataframe)
    dataframe = create_pir_features(dataframe)
    dataframe = create_co2_features(dataframe)

    selected_columns = [
        "timestamp",

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

        "Room_Occupancy_Count",
    ]

    return dataframe[selected_columns].copy()


# -------------------------------------------------------------------
# Save feature dataset
# -------------------------------------------------------------------

def save_features(dataframe):
    """
    Save the engineered feature dataset.
    """

    OUTPUT_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_DATA_PATH,
        index=False,
    )

    print(
        f"\nFeature dataset saved to:\n"
        f"{OUTPUT_DATA_PATH}"
    )


# -------------------------------------------------------------------
# Complete feature engineering pipeline
# -------------------------------------------------------------------

def engineer_features():
    """
    Run the complete feature engineering pipeline.

    Returns:
        pandas.DataFrame: Engineered dataset.
    """

    print("=" * 70)
    print("COGNITIVE IoT - FEATURE ENGINEERING")
    print("=" * 70)

    print("\nLoading cleaned dataset...")

    dataframe = load_cleaned_dataset()

    print(
        f"Loaded {len(dataframe)} rows "
        f"and {len(dataframe.columns)} columns."
    )

    print("\nCreating temperature features...")
    dataframe = create_temperature_features(dataframe)

    print("Creating light features...")
    dataframe = create_light_features(dataframe)

    print("Creating sound features...")
    dataframe = create_sound_features(dataframe)

    print("Creating PIR features...")
    dataframe = create_pir_features(dataframe)

    print("Creating CO2 features...")
    dataframe = create_co2_features(dataframe)

    print("\nSelecting final fused features...")

    dataframe = create_fused_features(dataframe)

    print(
        f"Final feature dataset shape: "
        f"{dataframe.shape[0]} rows × "
        f"{dataframe.shape[1]} columns"
    )

    print("\nFeatures:")
    for column in dataframe.columns:
        print(f"  - {column}")

    save_features(dataframe)

    print("\nFeature engineering complete.")

    return dataframe


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    engineer_features()