"""
Cognitive IoT - Adaptive Learning Engine

Processes feedback collected from the cognitive IoT pipeline.

The learning engine:

1. Loads recorded feedback.
2. Evaluates system performance.
3. Identifies incorrect decisions.
4. Analyses improvement cases.
5. Generates a learning dataset.
6. Checks whether enough validated feedback exists.
7. Combines validated feedback with the original UCI dataset.
8. Trains a candidate model.
9. Compares the candidate with the production model.
10. Promotes the candidate only when the safety criteria are met.

The production model is never replaced directly by feedback.
A candidate model is trained and evaluated first.
"""


import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ai.train_model import (
    RAW_FEATURES,
    TARGET,
    CONTEXT_FEATURES,
    FEATURE_FILE,
    create_context_features,
    create_model,
    fit_scaler,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEEDBACK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feedback_log.jsonl"
)

LEARNING_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learning_dataset.csv"
)

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

CANDIDATE_MODEL_FILE = (
    PROJECT_ROOT
    / "ai"
    / "model"
    / "candidate_occupancy_model.pkl"
)

CANDIDATE_SCALER_FILE = (
    PROJECT_ROOT
    / "ai"
    / "model"
    / "candidate_occupancy_scaler.pkl"
)


# ============================================================
# ADAPTIVE LEARNING CONFIGURATION
# ============================================================

MIN_FEEDBACK_SAMPLES = 50

VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

MINIMUM_ACCURACY_GAIN = 0.005


# ============================================================
# LEARNING ENGINE
# ============================================================

class LearningEngine:

    def __init__(
        self,
        feedback_file=FEEDBACK_FILE,
        learning_dataset=LEARNING_DATASET,
    ):

        self.feedback_file = Path(
            feedback_file
        )

        self.learning_dataset = Path(
            learning_dataset
        )

        self.learning_dataset.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # FEEDBACK LOADING
    # ========================================================

    def load_feedback_records(self):
        """
        Load all feedback records from the JSONL file.
        """

        if not self.feedback_file.exists():
            return []

        records = []

        with self.feedback_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    records.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError:

                    continue

        return records

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(self):
        """
        Calculate feedback and learning statistics.
        """

        records = (
            self.load_feedback_records()
        )

        total = len(records)

        correct = 0
        incorrect = 0
        pending = 0

        for record in records:

            feedback = record.get(
                "feedback"
            )

            if feedback == "correct":

                correct += 1

            elif feedback == "incorrect":

                incorrect += 1

            else:

                pending += 1

        evaluated = (
            correct + incorrect
        )

        if evaluated > 0:

            accuracy = (
                correct / evaluated
            )

        else:

            accuracy = 0.0

        return {

            "total_records":
                total,

            "evaluated_records":
                evaluated,

            "correct":
                correct,

            "incorrect":
                incorrect,

            "pending":
                pending,

            "feedback_accuracy":
                accuracy,
        }

    # ========================================================
    # EVALUATED RECORDS
    # ========================================================

    def get_evaluated_records(self):
        """
        Return records that have been explicitly evaluated.
        """

        records = (
            self.load_feedback_records()
        )

        evaluated_records = []

        for record in records:

            feedback = record.get(
                "feedback"
            )

            if feedback in (
                "correct",
                "incorrect",
            ):

                evaluated_records.append(
                    record
                )

        return evaluated_records

    # ========================================================
    # VALID TRAINING RECORDS
    # ========================================================

    def get_valid_training_records(self):
        """
        Return evaluated feedback records containing valid
        sensor data and a valid actual occupancy class.

        Only these records are allowed to influence adaptive
        training.
        """

        records = (
            self.get_evaluated_records()
        )

        valid_records = []

        for record in records:

            sensor_data = record.get(
                "sensor_data",
                {},
            )

            actual = record.get(
                "actual",
                {},
            )

            actual_class = actual.get(
                "occupancy_class"
            )

            required_sensor_values = [
                sensor_data.get(
                    "temperature"
                ),
                sensor_data.get(
                    "light"
                ),
                sensor_data.get(
                    "sound"
                ),
                sensor_data.get(
                    "co2"
                ),
                sensor_data.get(
                    "motion"
                ),
            ]

            if actual_class is None:
                continue

            if any(
                value is None
                for value in required_sensor_values
            ):
                continue

            try:

                actual_class = int(
                    actual_class
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if actual_class not in (
                0,
                1,
                2,
                3,
            ):
                continue

            valid_records.append(
                record
            )

        return valid_records

    # ========================================================
    # INCORRECT RECORDS
    # ========================================================

    def get_incorrect_records(self):
        """
        Return only incorrect evaluated decisions.
        """

        records = (
            self.get_evaluated_records()
        )

        return [

            record

            for record in records

            if record.get(
                "feedback"
            ) == "incorrect"

        ]

    # ========================================================
    # IMPROVEMENT ANALYSIS
    # ========================================================

    def analyse_improvement_cases(self):
        """
        Analyse incorrect predictions and extract useful
        information for future model improvement.
        """

        incorrect_records = (
            self.get_incorrect_records()
        )

        improvement_cases = []

        for record in incorrect_records:

            sensor_data = record.get(
                "sensor_data",
                {},
            )

            prediction = record.get(
                "prediction",
                {},
            )

            actual = record.get(
                "actual",
                {},
            )

            case = {

                "timestamp":
                    record.get(
                        "timestamp"
                    ),

                "predicted_class":
                    prediction.get(
                        "occupancy_class"
                    ),

                "predicted_state":
                    prediction.get(
                        "occupancy_state"
                    ),

                "confidence":
                    prediction.get(
                        "confidence"
                    ),

                "actual_class":
                    actual.get(
                        "occupancy_class"
                    ),

                "actual_state":
                    actual.get(
                        "occupancy_state"
                    ),

                "temperature":
                    sensor_data.get(
                        "temperature"
                    ),

                "co2":
                    sensor_data.get(
                        "co2"
                    ),

                "light":
                    sensor_data.get(
                        "light"
                    ),

                "sound":
                    sensor_data.get(
                        "sound"
                    ),

                "motion":
                    sensor_data.get(
                        "motion"
                    ),
            }

            improvement_cases.append(
                case
            )

        return improvement_cases

    # ========================================================
    # LEARNING DATASET
    # ========================================================

    def build_learning_dataset(self):
        """
        Convert evaluated records into a structured CSV
        learning dataset.
        """

        records = (
            self.get_evaluated_records()
        )

        rows = []

        for record in records:

            sensor_data = record.get(
                "sensor_data",
                {},
            )

            prediction = record.get(
                "prediction",
                {},
            )

            decision = record.get(
                "decision",
                {},
            )

            actual = record.get(
                "actual",
                {},
            )

            row = {

                "timestamp":
                    record.get(
                        "timestamp"
                    ),

                "temperature":
                    sensor_data.get(
                        "temperature"
                    ),

                "co2":
                    sensor_data.get(
                        "co2"
                    ),

                "light":
                    sensor_data.get(
                        "light"
                    ),

                "sound":
                    sensor_data.get(
                        "sound"
                    ),

                "motion":
                    sensor_data.get(
                        "motion"
                    ),

                "occupancy_class":
                    prediction.get(
                        "occupancy_class"
                    ),

                "occupancy_state":
                    prediction.get(
                        "occupancy_state"
                    ),

                "confidence":
                    prediction.get(
                        "confidence"
                    ),

                "actual_class":
                    actual.get(
                        "occupancy_class"
                    ),

                "actual_state":
                    actual.get(
                        "occupancy_state"
                    ),

                "cooling":
                    decision.get(
                        "cooling"
                    ),

                "ventilation":
                    decision.get(
                        "ventilation"
                    ),

                "lighting":
                    decision.get(
                        "lighting"
                    ),

                "priority":
                    decision.get(
                        "priority"
                    ),

                "feedback":
                    record.get(
                        "feedback"
                    ),
            }

            rows.append(row)

        if not rows:

            return pd.DataFrame()

        dataframe = pd.DataFrame(
            rows
        )

        dataframe.to_csv(
            self.learning_dataset,
            index=False,
        )

        return dataframe

    # ========================================================
    # FEEDBACK DATAFRAME
    # ========================================================

    def build_adaptive_feedback_data(self):
        """
        Convert validated feedback into the same raw feature
        format used by the original training pipeline.
        """

        records = (
            self.get_valid_training_records()
        )

        rows = []

        for record in records:

            sensor_data = record.get(
                "sensor_data",
                {},
            )

            actual = record.get(
                "actual",
                {},
            )

            rows.append({

                "temperature_mean":
                    float(
                        sensor_data[
                            "temperature"
                        ]
                    ),

                "light_mean":
                    float(
                        sensor_data[
                            "light"
                        ]
                    ),

                "sound_mean":
                    float(
                        sensor_data[
                            "sound"
                        ]
                    ),

                "co2":
                    float(
                        sensor_data[
                            "co2"
                        ]
                    ),

                "pir_activity":
                    float(
                        sensor_data[
                            "motion"
                        ]
                    ),

                TARGET:
                    int(
                        actual[
                            "occupancy_class"
                        ]
                    ),
            })

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # ORIGINAL DATASET
    # ========================================================

    def load_original_dataset(self):
        """
        Load the original UCI feature dataset.

        Only the exact raw features and target used by the
        production training pipeline are retained.
        """

        if not FEATURE_FILE.exists():

            raise FileNotFoundError(
                "Original feature dataset not found:\n"
                f"{FEATURE_FILE}"
            )

        dataframe = pd.read_csv(
            FEATURE_FILE
        )

        required_columns = (
            RAW_FEATURES + [TARGET]
        )

        missing_columns = [

            column

            for column in required_columns

            if column not in dataframe.columns

        ]

        if missing_columns:

            raise ValueError(
                "Original dataset is missing "
                "required columns:\n"
                + "\n".join(
                    missing_columns
                )
            )

        dataframe = dataframe[
            required_columns
        ].dropna().copy()

        dataframe[TARGET] = (
            dataframe[TARGET]
            .astype(int)
        )

        return dataframe

    # ========================================================
    # BUILD CANDIDATE DATA
    # ========================================================

    def build_candidate_training_data(self):
        """
        Split the original UCI dataset into a stable training
        and validation portion, then combine validated feedback
        with the original training portion.

        The original UCI validation portion remains untouched.
        """

        original_data = (
            self.load_original_dataset()
        )

        feedback_data = (
            self.build_adaptive_feedback_data()
        )

        original_train, original_validation = (
            train_test_split(
                original_data,
                test_size=VALIDATION_SIZE,
                random_state=RANDOM_STATE,
                stratify=original_data[TARGET],
            )
        )

        if len(feedback_data) > 0:

            feedback_train = (
                feedback_data
            )

        else:

            feedback_train = (
                pd.DataFrame(
                    columns=original_data.columns
                )
            )

        candidate_training_data = (
            pd.concat(
                [
                    original_train,
                    feedback_train,
                ],
                ignore_index=True,
            )
        )

        return (
            candidate_training_data,
            original_validation,
        )

    # ========================================================
    # PRODUCTION MODEL
    # ========================================================

    def load_production_model(self):
        """
        Load the current production model and scaler.
        """

        if not MODEL_FILE.exists():

            raise FileNotFoundError(
                f"Production model not found:\n"
                f"{MODEL_FILE}"
            )

        if not SCALER_FILE.exists():

            raise FileNotFoundError(
                f"Production scaler not found:\n"
                f"{SCALER_FILE}"
            )

        model = joblib.load(
            MODEL_FILE
        )

        scaler = joblib.load(
            SCALER_FILE
        )

        return (
            model,
            scaler,
        )

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    def evaluate_model_on_data(
        self,
        model,
        scaler,
        dataframe,
    ):
        """
        Evaluate a model using the same context-feature
        pipeline as the production system.
        """

        X = dataframe[
            RAW_FEATURES
        ]

        y = dataframe[
            TARGET
        ]

        context = (
            create_context_features(
                X,
                scaler,
            )
        )

        predictions = model.predict(
            context[
                CONTEXT_FEATURES
            ]
        )

        return accuracy_score(
            y,
            predictions,
        )

    # ========================================================
    # TRAIN CANDIDATE
    # ========================================================

    def train_candidate_model(
        self,
        training_data,
    ):
        """
        Train a candidate model using:

        Original UCI training data
                    +
        Validated feedback data
        """

        X = training_data[
            RAW_FEATURES
        ].copy()

        y = training_data[
            TARGET
        ].astype(int).copy()

        scaler = fit_scaler(
            X
        )

        train_context = (
            create_context_features(
                X,
                scaler,
            )
        )

        model = create_model()

        model.fit(
            train_context[
                CONTEXT_FEATURES
            ],
            y,
        )

        return (
            model,
            scaler,
        )

    # ========================================================
    # PROMOTION DECISION
    # ========================================================

    def should_promote_candidate(
        self,
        production_accuracy,
        candidate_accuracy,
    ):
        """
        A candidate must improve validation accuracy by at
        least the configured minimum gain.
        """

        improvement = (
            candidate_accuracy
            - production_accuracy
        )

        return (
            improvement
            >= MINIMUM_ACCURACY_GAIN
        )

    # ========================================================
    # SAVE CANDIDATE
    # ========================================================

    def save_candidate_model(
        self,
        model,
        scaler,
    ):
        """
        Save the candidate separately from production.
        """

        joblib.dump(
            model,
            CANDIDATE_MODEL_FILE,
        )

        joblib.dump(
            scaler,
            CANDIDATE_SCALER_FILE,
        )

    # ========================================================
    # PROMOTE CANDIDATE
    # ========================================================

    def promote_candidate_model(self):
        """
        Promote the validated candidate to production.
        """

        if not CANDIDATE_MODEL_FILE.exists():

            raise FileNotFoundError(
                "Candidate model not found."
            )

        if not CANDIDATE_SCALER_FILE.exists():

            raise FileNotFoundError(
                "Candidate scaler not found."
            )

        candidate_model = joblib.load(
            CANDIDATE_MODEL_FILE
        )

        candidate_scaler = joblib.load(
            CANDIDATE_SCALER_FILE
        )

        joblib.dump(
            candidate_model,
            MODEL_FILE,
        )

        joblib.dump(
            candidate_scaler,
            SCALER_FILE,
        )

    # ========================================================
    # ADAPTIVE TRAINING
    # ========================================================

    def run_adaptive_training(self):
        """
        Execute controlled adaptive training.

        The candidate uses the original UCI training data plus
        validated feedback.

        The original UCI validation set is kept untouched for
        a fair comparison between the production and candidate
        models.
        """

        feedback_data = (
            self.build_adaptive_feedback_data()
        )

        sample_count = len(
            feedback_data
        )

        # ----------------------------------------------------
        # Check feedback quantity
        # ----------------------------------------------------

        if sample_count < MIN_FEEDBACK_SAMPLES:

            return {

                "status":
                    "insufficient_feedback",

                "message":
                    (
                        "Not enough validated feedback "
                        "samples for adaptive retraining."
                    ),

                "samples":
                    sample_count,

                "required":
                    MIN_FEEDBACK_SAMPLES,
            }

        # ----------------------------------------------------
        # Check class diversity
        # ----------------------------------------------------

        unique_classes = (
            feedback_data[
                TARGET
            ].nunique()
        )

        if unique_classes < 2:

            return {

                "status":
                    "insufficient_classes",

                "message":
                    (
                        "Validated feedback does not "
                        "contain enough occupancy classes."
                    ),

                "samples":
                    sample_count,

                "classes":
                    unique_classes,
            }

        # ----------------------------------------------------
        # Build candidate training data
        # ----------------------------------------------------

        (
            candidate_training_data,
            validation_data,
        ) = (
            self.build_candidate_training_data()
        )

        # ----------------------------------------------------
        # Load production model
        # ----------------------------------------------------

        (
            production_model,
            production_scaler,
        ) = (
            self.load_production_model()
        )

        # ----------------------------------------------------
        # Evaluate current production model
        # ----------------------------------------------------

        production_accuracy = (
            self.evaluate_model_on_data(
                production_model,
                production_scaler,
                validation_data,
            )
        )

        # ----------------------------------------------------
        # Train candidate
        # ----------------------------------------------------

        (
            candidate_model,
            candidate_scaler,
        ) = (
            self.train_candidate_model(
                candidate_training_data
            )
        )

        # ----------------------------------------------------
        # Evaluate candidate
        # ----------------------------------------------------

        candidate_accuracy = (
            self.evaluate_model_on_data(
                candidate_model,
                candidate_scaler,
                validation_data,
            )
        )

        # ----------------------------------------------------
        # Save candidate
        # ----------------------------------------------------

        self.save_candidate_model(
            candidate_model,
            candidate_scaler,
        )

        # ----------------------------------------------------
        # Promotion decision
        # ----------------------------------------------------

        promote = (
            self.should_promote_candidate(
                production_accuracy,
                candidate_accuracy,
            )
        )

        if promote:

            self.promote_candidate_model()

            status = (
                "candidate_promoted"
            )

            message = (
                "Candidate model passed "
                "the promotion criteria."
            )

        else:

            status = (
                "candidate_rejected"
            )

            message = (
                "Candidate model did not "
                "meet the promotion criteria. "
                "Production model retained."
            )

        return {

            "status":
                status,

            "message":
                message,

            "samples":
                sample_count,

            "candidate_training_rows":
                len(
                    candidate_training_data
                ),

            "validation_rows":
                len(
                    validation_data
                ),

            "production_accuracy":
                production_accuracy,

            "candidate_accuracy":
                candidate_accuracy,

            "accuracy_gain":
                (
                    candidate_accuracy
                    - production_accuracy
                ),

            "promoted":
                promote,
        }

    # ========================================================
    # LEARNING SUMMARY
    # ========================================================

    def get_learning_summary(self):
        """
        Generate a summary of the current learning state.
        """

        statistics = (
            self.get_statistics()
        )

        improvement_cases = (
            self.analyse_improvement_cases()
        )

        valid_training_records = (
            self.get_valid_training_records()
        )

        if statistics[
            "evaluated_records"
        ] == 0:

            learning_status = (
                "Waiting for feedback"
            )

        elif len(
            valid_training_records
        ) < MIN_FEEDBACK_SAMPLES:

            learning_status = (
                "Collecting feedback "
                "for adaptive learning"
            )

        elif statistics[
            "incorrect"
        ] > 0:

            learning_status = (
                "Adaptive training available"
            )

        else:

            learning_status = (
                "Validated feedback available"
            )

        return {

            "learning_status":
                learning_status,

            "total_records":
                statistics[
                    "total_records"
                ],

            "evaluated_records":
                statistics[
                    "evaluated_records"
                ],

            "valid_training_records":
                len(
                    valid_training_records
                ),

            "required_training_records":
                MIN_FEEDBACK_SAMPLES,

            "correct_decisions":
                statistics[
                    "correct"
                ],

            "incorrect_decisions":
                statistics[
                    "incorrect"
                ],

            "pending_decisions":
                statistics[
                    "pending"
                ],

            "feedback_accuracy":
                statistics[
                    "feedback_accuracy"
                ],

            "improvement_cases":
                len(
                    improvement_cases
                ),
        }

    # ========================================================
    # COMPLETE LEARNING CYCLE
    # ========================================================

    def run_learning_cycle(self):
        """
        Execute one complete learning cycle.
        """

        statistics = (
            self.get_statistics()
        )

        learning_dataset = (
            self.build_learning_dataset()
        )

        improvement_cases = (
            self.analyse_improvement_cases()
        )

        summary = (
            self.get_learning_summary()
        )

        adaptive_result = (
            self.run_adaptive_training()
        )

        return {

            "statistics":
                statistics,

            "learning_dataset_rows":
                len(
                    learning_dataset
                ),

            "improvement_cases":
                improvement_cases,

            "summary":
                summary,

            "adaptive_training":
                adaptive_result,
        }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    engine = LearningEngine()

    result = (
        engine.run_learning_cycle()
    )

    print()

    print(
        "========================================"
    )

    print(
        "       COGNITIVE IoT LEARNING ENGINE"
    )

    print(
        "========================================"
    )

    print()

    summary = result[
        "summary"
    ]

    statistics = result[
        "statistics"
    ]

    print(
        "Learning Status :",
        summary[
            "learning_status"
        ],
    )

    print(
        "Total Records   :",
        statistics[
            "total_records"
        ],
    )

    print(
        "Evaluated       :",
        statistics[
            "evaluated_records"
        ],
    )

    print(
        "Valid Training  :",
        summary[
            "valid_training_records"
        ],
    )

    print(
        "Required        :",
        summary[
            "required_training_records"
        ],
    )

    print(
        "Correct         :",
        statistics[
            "correct"
        ],
    )

    print(
        "Incorrect       :",
        statistics[
            "incorrect"
        ],
    )

    print(
        "Pending         :",
        statistics[
            "pending"
        ],
    )

    print(
        "Feedback Accuracy:",
        f"{statistics['feedback_accuracy']:.2%}",
    )

    print(
        "Learning Dataset:",
        result[
            "learning_dataset_rows"
        ],
        "rows",
    )

    print()

    print(
        "Improvement Cases:"
    )

    print(
        "------------------"
    )

    improvement_cases = result[
        "improvement_cases"
    ]

    if not improvement_cases:

        print(
            "No incorrect decisions detected."
        )

    else:

        for number, case in enumerate(
            improvement_cases,
            start=1,
        ):

            print()

            print(
                f"Case {number}"
            )

            print(
                "Predicted :",
                case[
                    "predicted_state"
                ],
            )

            confidence = (
                case[
                    "confidence"
                ]
            )

            if confidence is not None:

                print(
                    "Confidence:",
                    f"{confidence:.2%}",
                )

            else:

                print(
                    "Confidence: N/A"
                )

            print(
                "Actual    :",
                case[
                    "actual_state"
                ],
            )

            print(
                "Temperature:",
                case[
                    "temperature"
                ],
            )

            print(
                "CO2        :",
                case[
                    "co2"
                ],
            )

            print(
                "Light      :",
                case[
                    "light"
                ],
            )

            print(
                "Sound      :",
                case[
                    "sound"
                ],
            )

            print(
                "Motion     :",
                case[
                    "motion"
                ],
            )

            print(
                "Action     :",
                "Flagged for model improvement",
            )

    print()

    print(
        "Adaptive Training:"
    )

    print(
        "------------------"
    )

    adaptive = result[
        "adaptive_training"
    ]

    print(
        "Status  :",
        adaptive[
            "status"
        ],
    )

    print(
        "Message :",
        adaptive[
            "message"
        ],
    )

    if (
        adaptive[
            "status"
        ]
        == "insufficient_feedback"
    ):

        print(
            "Samples :",
            adaptive[
                "samples"
            ],
            "/",
            adaptive[
                "required"
            ],
        )

    elif (
        adaptive[
            "status"
        ]
        in (
            "candidate_promoted",
            "candidate_rejected",
        )
    ):

        print(
            "Samples                :",
            adaptive[
                "samples"
            ],
        )

        print(
            "Candidate training rows:",
            adaptive[
                "candidate_training_rows"
            ],
        )

        print(
            "Validation rows        :",
            adaptive[
                "validation_rows"
            ],
        )

        print(
            "Production Accuracy    :",
            f"{adaptive['production_accuracy']:.2%}",
        )

        print(
            "Candidate Accuracy     :",
            f"{adaptive['candidate_accuracy']:.2%}",
        )

        print(
            "Accuracy Gain          :",
            f"{adaptive['accuracy_gain']:.2%}",
        )

        print(
            "Promoted               :",
            adaptive[
                "promoted"
            ],
        )

    print()

    print(
        "Learning cycle completed."
    )