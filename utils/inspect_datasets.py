"""
Dataset inspection utility for the Cognitive IoT project.

This script provides a basic statistical overview of the two
real-world occupancy datasets used by the project.
"""

from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
ROOM_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "room_occupancy_estimation"
    / "Occupancy_Estimation.csv"
)

OCCUPANCY_DATASET = PROJECT_ROOT / "dataset" / "occupancy_detection"


def inspect_dataframe(name, dataframe):
    """Print important information about a dataset."""

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    print(f"\nRows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    print("\nColumn names:")
    for column in dataframe.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(dataframe.dtypes)

    print("\nMissing values:")
    missing = dataframe.isnull().sum()
    print(missing)

    print("\nFirst 5 rows:")
    print(dataframe.head())

    print("\nBasic statistics:")
    print(dataframe.describe(include="all").transpose())


def inspect_room_occupancy():
    """Inspect the Room Occupancy Estimation dataset."""

    dataframe = pd.read_csv(ROOM_DATASET)

    inspect_dataframe(
        "UCI Room Occupancy Estimation",
        dataframe,
    )

    print("\nRoom Occupancy Count distribution:")
    print(dataframe["Room_Occupancy_Count"].value_counts().sort_index())

    print("\nDate range:")
    print(f"Start: {dataframe['Date'].min()}")
    print(f"End:   {dataframe['Date'].max()}")


def inspect_occupancy_detection():
    """Inspect the UCI Occupancy Detection dataset."""

    files = [
        "datatraining.txt",
        "datatest.txt",
        "datatest2.txt",
    ]

    for filename in files:
        file_path = OCCUPANCY_DATASET / filename

        dataframe = pd.read_csv(file_path)

        inspect_dataframe(
            f"UCI Occupancy Detection - {filename}",
            dataframe,
        )

        print("\nOccupancy distribution:")
        print(dataframe["Occupancy"].value_counts().sort_index())


def main():
    """Run inspection of both datasets."""

    print("Cognitive IoT - Dataset Inspection")

    inspect_room_occupancy()
    inspect_occupancy_detection()

    print("\n" + "=" * 70)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()