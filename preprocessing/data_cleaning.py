"""
Data cleaning module for the Cognitive IoT project.

This module loads the real UCI Room Occupancy Estimation dataset,
performs basic validation and cleaning, and prepares it for
feature engineering.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "room_occupancy_estimation"
    / "Occupancy_Estimation.csv"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "room_occupancy_cleaned.csv"


# -------------------------------------------------------------------
# Expected columns
# -------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "Date",
    "Time",
    "S1_Temp",
    "S2_Temp",
    "S3_Temp",
    "S4_Temp",
    "S1_Light",
    "S2_Light",
    "S3_Light",
    "S4_Light",
    "S1_Sound",
    "S2_Sound",
    "S3_Sound",
    "S4_Sound",
    "S5_CO2",
    "S5_CO2_Slope",
    "S6_PIR",
    "S7_PIR",
    "Room_Occupancy_Count",
]


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

def load_dataset():
    """
    Load the Room Occupancy Estimation dataset.

    Returns:
        pandas.DataFrame: Raw dataset.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    dataframe = pd.read_csv(RAW_DATA_PATH)

    return dataframe


# -------------------------------------------------------------------
# Validate columns
# -------------------------------------------------------------------

def validate_columns(dataframe):
    """
    Check whether all expected columns are present.

    Parameters:
        dataframe (pandas.DataFrame): Dataset to validate.

    Raises:
        ValueError: If required columns are missing.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )


# -------------------------------------------------------------------
# Clean timestamps
# -------------------------------------------------------------------

def clean_timestamp(dataframe):
    """
    Combine Date and Time into a single timestamp column.

    Parameters:
        dataframe (pandas.DataFrame): Dataset.

    Returns:
        pandas.DataFrame: Dataset with timestamp column.
    """

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["Date"] + " " + dataframe["Time"],
        errors="coerce",
    )

    invalid_timestamps = dataframe["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        print(
            f"Warning: {invalid_timestamps} invalid timestamps found."
        )

    return dataframe


# -------------------------------------------------------------------
# Handle missing values
# -------------------------------------------------------------------

def handle_missing_values(dataframe):
    """
    Check and handle missing values.

    Since the original dataset is expected to contain complete
    observations, rows with missing values are removed only if
    they are actually present.

    Parameters:
        dataframe (pandas.DataFrame): Dataset.

    Returns:
        pandas.DataFrame: Cleaned dataset.
    """

    dataframe = dataframe.copy()

    missing_before = dataframe.isnull().sum().sum()

    if missing_before == 0:
        print("No missing values found.")
        return dataframe

    print(f"Missing values found: {missing_before}")
    print("Removing rows containing missing values.")

    dataframe = dataframe.dropna().reset_index(drop=True)

    missing_after = dataframe.isnull().sum().sum()

    print(f"Missing values after cleaning: {missing_after}")

    return dataframe


# -------------------------------------------------------------------
# Validate sensor values
# -------------------------------------------------------------------

def validate_sensor_values(dataframe):
    """
    Check for obviously invalid sensor values.

    This function performs basic validation without aggressively
    removing unusual observations.

    Parameters:
        dataframe (pandas.DataFrame): Dataset.

    Returns:
        pandas.DataFrame: Validated dataset.
    """

    dataframe = dataframe.copy()

    sensor_columns = [
        "S1_Temp",
        "S2_Temp",
        "S3_Temp",
        "S4_Temp",
        "S1_Light",
        "S2_Light",
        "S3_Light",
        "S4_Light",
        "S1_Sound",
        "S2_Sound",
        "S3_Sound",
        "S4_Sound",
        "S5_CO2",
        "S5_CO2_Slope",
        "S6_PIR",
        "S7_PIR",
    ]

    print("\nSensor validation:")

    for column in sensor_columns:
        invalid_count = dataframe[column].isna().sum()

        print(
            f"  {column}: "
            f"{invalid_count} missing values"
        )

    return dataframe


# -------------------------------------------------------------------
# Remove unnecessary columns
# -------------------------------------------------------------------

def select_columns(dataframe):
    """
    Select the columns required by the project.

    Date and Time are retained temporarily for traceability,
    while timestamp becomes the main temporal representation.

    Parameters:
        dataframe (pandas.DataFrame): Dataset.

    Returns:
        pandas.DataFrame: Selected dataset.
    """

    selected_columns = [
        "timestamp",
        "S1_Temp",
        "S2_Temp",
        "S3_Temp",
        "S4_Temp",
        "S1_Light",
        "S2_Light",
        "S3_Light",
        "S4_Light",
        "S1_Sound",
        "S2_Sound",
        "S3_Sound",
        "S4_Sound",
        "S5_CO2",
        "S5_CO2_Slope",
        "S6_PIR",
        "S7_PIR",
        "Room_Occupancy_Count",
    ]

    return dataframe[selected_columns].copy()


# -------------------------------------------------------------------
# Save cleaned dataset
# -------------------------------------------------------------------

def save_dataset(dataframe):
    """
    Save the cleaned dataset to the processed-data directory.

    Parameters:
        dataframe (pandas.DataFrame): Cleaned dataset.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        CLEANED_DATA_PATH,
        index=False,
    )

    print(
        f"\nCleaned dataset saved to:\n"
        f"{CLEANED_DATA_PATH}"
    )


# -------------------------------------------------------------------
# Complete cleaning pipeline
# -------------------------------------------------------------------

def clean_dataset():
    """
    Run the complete data-cleaning pipeline.

    Returns:
        pandas.DataFrame: Cleaned dataset.
    """

    print("=" * 70)
    print("COGNITIVE IoT - DATA CLEANING")
    print("=" * 70)

    print("\nLoading dataset...")

    dataframe = load_dataset()

    print(
        f"Loaded {len(dataframe)} rows "
        f"and {len(dataframe.columns)} columns."
    )

    print("\nValidating columns...")

    validate_columns(dataframe)

    print("All expected columns are present.")

    print("\nCreating timestamp...")

    dataframe = clean_timestamp(dataframe)

    print("Timestamp created successfully.")

    print("\nChecking missing values...")

    dataframe = handle_missing_values(dataframe)

    dataframe = validate_sensor_values(dataframe)

    print("\nSelecting project columns...")

    dataframe = select_columns(dataframe)

    print(
        f"Final dataset shape: "
        f"{dataframe.shape[0]} rows × "
        f"{dataframe.shape[1]} columns"
    )

    save_dataset(dataframe)

    print("\nData cleaning complete.")

    return dataframe


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    clean_dataset()