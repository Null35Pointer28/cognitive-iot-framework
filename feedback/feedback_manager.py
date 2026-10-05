"""
Cognitive IoT - Feedback Manager

Collects system decisions and feedback so that the cognitive
system can evaluate its decisions and build learning data.
"""

import json
from datetime import datetime
from pathlib import Path


FEEDBACK_FILE = Path("data/processed/feedback_log.jsonl")


class FeedbackManager:

    def __init__(self, feedback_file=FEEDBACK_FILE):

        self.feedback_file = Path(feedback_file)

        self.feedback_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Convert DecisionResult into JSON-safe dictionary
    # --------------------------------------------------------

    def _decision_to_dict(self, decision):

        if isinstance(decision, dict):
            return decision

        return {
            "occupancy_state": getattr(
                decision,
                "occupancy_state",
                None
            ),

            "occupancy_confidence": getattr(
                decision,
                "occupancy_confidence",
                None
            ),

            "cooling": getattr(
                decision,
                "cooling",
                None
            ),

            "ventilation": getattr(
                decision,
                "ventilation",
                None
            ),

            "lighting": getattr(
                decision,
                "lighting",
                None
            ),

            "priority": getattr(
                decision,
                "priority",
                None
            ),

            "reasons": getattr(
                decision,
                "reasons",
                []
            ),
        }

    # --------------------------------------------------------
    # Record feedback
    # --------------------------------------------------------

    def record_feedback(
        self,
        sensor_data,
        prediction,
        decision,
        feedback=None,
    ):
        """
        Record one cognitive decision and its feedback.

        feedback can be:
        - None
        - "correct"
        - "incorrect"
        """

        decision_data = self._decision_to_dict(
            decision
        )

        record = {
            "timestamp": datetime.now().isoformat(),

            "sensor_data": sensor_data,

            "prediction": prediction,

            "decision": decision_data,

            "feedback": feedback,
        }

        with self.feedback_file.open(
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(record) + "\n"
            )

    # --------------------------------------------------------
    # Load feedback records
    # --------------------------------------------------------

    def get_feedback_records(self):
        """
        Load all recorded feedback entries.
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

    # --------------------------------------------------------
    # Feedback statistics
    # --------------------------------------------------------

    def get_statistics(self):
        """
        Calculate basic feedback statistics.
        """

        records = self.get_feedback_records()

        total = len(records)

        correct = sum(
            1
            for record in records
            if record.get("feedback") == "correct"
        )

        incorrect = sum(
            1
            for record in records
            if record.get("feedback") == "incorrect"
        )

        pending = sum(
            1
            for record in records
            if record.get("feedback") is None
        )

        evaluated = correct + incorrect

        accuracy = (
            correct / evaluated
            if evaluated > 0
            else 0.0
        )

        return {
            "total_records": total,
            "correct": correct,
            "incorrect": incorrect,
            "pending": pending,
            "feedback_accuracy": accuracy,
        }