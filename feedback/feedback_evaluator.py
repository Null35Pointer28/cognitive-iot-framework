"""
Cognitive IoT - Feedback Evaluator

Evaluates a cognitive decision against the actual observed
occupancy condition and updates the corresponding feedback
record.

This creates the bridge between:

AI Prediction → Real/Observed Condition → Feedback → Learning
"""

import json
from pathlib import Path


FEEDBACK_FILE = Path(
    "data/processed/feedback_log.jsonl"
)


class FeedbackEvaluator:

    def __init__(
        self,
        feedback_file=FEEDBACK_FILE,
    ):

        self.feedback_file = Path(
            feedback_file
        )

    def load_records(self):
        """
        Load all feedback records.
        """

        if not self.feedback_file.exists():
            return []

        records = []

        with self.feedback_file.open(
            "r",
            encoding="utf-8"
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

    def save_records(self, records):
        """
        Rewrite the feedback file with
        updated records.
        """

        self.feedback_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.feedback_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            for record in records:

                file.write(
                    json.dumps(record)
                    + "\n"
                )

    def evaluate_latest(
        self,
        actual_occupancy,
    ):
        """
        Evaluate the latest pending decision.

        Parameters
        ----------
        actual_occupancy : int
            Actual occupancy class.

            0 = Empty
            1 = Low Occupancy
            2 = Medium Occupancy
            3 = High Occupancy
        """

        valid_classes = {
            0: "Empty",
            1: "Low Occupancy",
            2: "Medium Occupancy",
            3: "High Occupancy",
        }

        if actual_occupancy not in valid_classes:

            raise ValueError(
                "actual_occupancy must be "
                "0, 1, 2, or 3."
            )

        records = self.load_records()

        if not records:

            return {
                "success": False,
                "message": "No feedback records found."
            }

        pending_index = None

        # Find the most recent pending record.
        for index in range(
            len(records) - 1,
            -1,
            -1
        ):

            if records[index].get(
                "feedback"
            ) is None:

                pending_index = index
                break

        if pending_index is None:

            return {
                "success": False,
                "message":
                    "No pending feedback records found."
            }

        record = records[pending_index]

        prediction = record.get(
            "prediction",
            {}
        )

        predicted_class = prediction.get(
            "occupancy_class"
        )

        predicted_state = prediction.get(
            "occupancy_state",
            "Unknown"
        )

        actual_state = valid_classes[
            actual_occupancy
        ]

        is_correct = (
            predicted_class
            == actual_occupancy
        )

        if is_correct:
            feedback = "correct"
        else:
            feedback = "incorrect"

        record["feedback"] = feedback

        record["actual"] = {
            "occupancy_class":
                actual_occupancy,
            "occupancy_state":
                actual_state,
        }

        self.save_records(records)

        return {
            "success": True,
            "feedback": feedback,
            "predicted_class":
                predicted_class,
            "predicted_state":
                predicted_state,
            "actual_class":
                actual_occupancy,
            "actual_state":
                actual_state,
        }


if __name__ == "__main__":

    evaluator = FeedbackEvaluator()

    print()
    print(
        "========================================"
    )
    print(
        "       COGNITIVE IoT FEEDBACK"
    )
    print(
        "           EVALUATOR"
    )
    print(
        "========================================"
    )

    print()

    print(
        "Occupancy classes:"
    )

    print(
        "0 = Empty"
    )

    print(
        "1 = Low Occupancy"
    )

    print(
        "2 = Medium Occupancy"
    )

    print(
        "3 = High Occupancy"
    )

    print()

    user_input = input(
        "Enter actual occupancy "
        "(0-3): "
    )

    try:

        actual_occupancy = int(
            user_input
        )

        result = (
            evaluator.evaluate_latest(
                actual_occupancy
            )
        )

        print()

        if not result["success"]:

            print(
                "Feedback Error:",
                result["message"]
            )

        else:

            print(
                "AI Prediction :",
                result[
                    "predicted_state"
                ]
            )

            print(
                "Actual State  :",
                result[
                    "actual_state"
                ]
            )

            print(
                "Feedback      :",
                result[
                    "feedback"
                ].upper()
            )

            print()

            if result["feedback"] == "correct":

                print(
                    "Learning Engine:"
                )

                print(
                    "Decision recorded "
                    "as successful."
                )

            else:

                print(
                    "Learning Engine:"
                )

                print(
                    "Improvement case "
                    "detected."
                )

    except ValueError:

        print(
            "Invalid input."
        )