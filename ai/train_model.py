"""
Cognitive IoT - Production Model Training

Provides the single source of truth for the occupancy classification
training pipeline.

Pipeline:
Raw Features
    ↓
Train/Test Split
    ↓
Fit Scaler on Training Data Only
    ↓
Create Context Features
    ↓
Random Forest Classifier
    ↓
Evaluate
    ↓
Save Model + Scaler

The reusable functions in this module are also intended to be used
by the adaptive learning engine for future candidate-model training.
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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_occupancy_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ai"
    / "model"
)

MODEL_FILE = MODEL_DIR / "occupancy_model.pkl"
SCALER_FILE = MODEL_DIR / "occupancy_scaler.pkl"


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

TARGET = "Room_Occupancy_Count"

CONTEXT_FEATURES = [
    "temperature_context",
    "lighting_context",
    "sound_context",
    "co2_context",
    "motion_context",
    "unified_context_score",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():
    """
    Load and validate the processed feature dataset.

    Returns:
        pandas.DataFrame
    """

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    dataframe = pd.read_csv(FEATURE_FILE)

    required_columns = RAW_FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    dataframe = dataframe.dropna(
        subset=required_columns
    ).copy()

    return dataframe


# ============================================================
# CONTEXT FEATURE CREATION
# ============================================================

def create_context_features(
    dataframe,
    scaler,
):
    """
    Convert raw sensor features into normalized cognitive
    context features.

    The scaler must already be fitted on training data.
    """

    scaled_values = scaler.transform(
        dataframe[RAW_FEATURES]
    )

    scaled = pd.DataFrame(
        scaled_values,
        columns=RAW_FEATURES,
        index=dataframe.index,
    )

    context = pd.DataFrame(
        index=dataframe.index
    )

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

    return context


# ============================================================
# SCALER
# ============================================================

def fit_scaler(X_train):
    """
    Fit the MinMaxScaler using training data only.

    Returns:
        MinMaxScaler
    """

    scaler = MinMaxScaler()

    scaler.fit(
        X_train[RAW_FEATURES]
    )

    return scaler


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():
    """
    Create the production Random Forest classifier.

    Keeping model creation in one function ensures that future
    adaptive training uses the same model configuration.
    """

    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    X_train,
    y_train,
    scaler,
):
    """
    Train a Random Forest model using the project's cognitive
    context feature pipeline.

    Args:
        X_train: Raw training features.
        y_train: Training labels.
        scaler: Fitted MinMaxScaler.

    Returns:
        Trained RandomForestClassifier
    """

    train_context = create_context_features(
        X_train,
        scaler,
    )

    model = create_model()

    model.fit(
        train_context[CONTEXT_FEATURES],
        y_train,
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    scaler,
    verbose=True,
):
    """
    Evaluate a trained model using the same context-feature
    pipeline used during training.

    Returns:
        Dictionary containing predictions, accuracy,
        classification report and confusion matrix.
    """

    test_context = create_context_features(
        X_test,
        scaler,
    )

    predictions = model.predict(
        test_context[CONTEXT_FEATURES]
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    if verbose:
        print(
            f"\nAccuracy: {accuracy * 100:.4f}%"
        )

        print(
            "\nClassification Report:"
        )

        print(report)

        print(
            "Confusion Matrix:"
        )

        print(matrix)

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model):
    """
    Return feature importance values from the trained model.
    """

    importance = pd.Series(
        model.feature_importances_,
        index=CONTEXT_FEATURES,
    ).sort_values(
        ascending=False
    )

    return importance


# ============================================================
# MODEL SAVING
# ============================================================

def save_model(
    model,
    scaler,
):
    """
    Save the trained model and scaler to the production
    model directory.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        scaler,
        SCALER_FILE,
    )


# ============================================================
# MAIN PRODUCTION TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 50)
    print("      COGNITIVE IoT MODEL TRAINING")
    print("=" * 50)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n[1] Loading dataset")

    dataframe = load_training_data()

    print(
        f"Dataset rows : {len(dataframe)}"
    )

    print(
        f"Dataset cols : {len(dataframe.columns)}"
    )

    print(
        f"Rows after cleaning : {len(dataframe)}"
    )

    # --------------------------------------------------------
    # 2. Prepare X and y
    # --------------------------------------------------------

    X = dataframe[RAW_FEATURES].copy()

    y = dataframe[TARGET].astype(int).copy()

    print("\nOccupancy distribution:")

    for class_value, count in (
        y.value_counts()
        .sort_index()
        .items()
    ):
        print(
            f"  Class {class_value}: {count}"
        )

    # --------------------------------------------------------
    # 3. Stratified train/test split
    # --------------------------------------------------------

    print(
        "\n[2] Creating train/test split"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    # --------------------------------------------------------
    # 4. Fit scaler ONLY on training data
    # --------------------------------------------------------

    print(
        "\n[3] Fitting scaler on training data only"
    )

    scaler = fit_scaler(
        X_train
    )

    # --------------------------------------------------------
    # 5. Create context features
    # --------------------------------------------------------

    print(
        "\n[4] Creating cognitive context features"
    )

    train_context = create_context_features(
        X_train,
        scaler,
    )

    print(
        "Context features:"
    )

    for feature in CONTEXT_FEATURES:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # 6. Train Random Forest
    # --------------------------------------------------------

    print(
        "\n[5] Training Random Forest"
    )

    model = create_model()

    model.fit(
        train_context[CONTEXT_FEATURES],
        y_train,
    )

    # --------------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------------

    print(
        "\n[6] Evaluating model"
    )

    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        verbose=True,
    )

    # --------------------------------------------------------
    # 8. Feature importance
    # --------------------------------------------------------

    print(
        "\nFeature Importance:"
    )

    importance = get_feature_importance(
        model
    )

    for feature, value in importance.items():
        print(
            f"  {feature:<25} {value:.4f}"
        )

    # --------------------------------------------------------
    # 9. Save model and scaler
    # --------------------------------------------------------

    print(
        "\n[7] Saving model and scaler"
    )

    save_model(
        model=model,
        scaler=scaler,
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )

    print(
        f"\nScaler saved to:"
        f"\n{SCALER_FILE}"
    )

    print(
        "\nTraining completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()