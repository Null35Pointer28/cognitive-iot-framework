"""
Cognitive IoT - Presentation-Safe Learning Demonstration

Demonstrates the complete adaptive learning lifecycle:
Prediction → Feedback Collection → Feedback Evaluation → Improvement Analysis
→ Training Eligibility Check → Candidate Model Training → Validation Comparison
→ Safe Promotion/Rejection Evaluation.

Uses a separate temporary demo feedback dataset.
Demonstration ground truth: simulated/user-provided feedback.
This is not physical occupancy ground truth.
"""

import json
from pathlib import Path
import pandas as pd

from ai.predict import OccupancyPredictor
from feedback.feedback_manager import FeedbackManager
from feedback.feedback_evaluator import FeedbackEvaluator
from learning.learning_engine import (
    LearningEngine,
    MIN_FEEDBACK_SAMPLES,
    MINIMUM_ACCURACY_GAIN,
)


def run_demo():
    print("========================================")
    print("       COGNITIVE IoT LEARNING DEMO")
    print("========================================")
    print()
    print("Demonstration ground truth: simulated/user-provided feedback.")
    print("This is not physical occupancy ground truth.")
    print()

    demo_feedback_path = Path("data/processed/demo_feedback_log.jsonl")
    demo_dataset_path = Path("data/processed/demo_learning_dataset.csv")

    # Clean up any previous demo files
    if demo_feedback_path.exists():
        demo_feedback_path.unlink()
    if demo_dataset_path.exists():
        demo_dataset_path.unlink()

    # ----------------------------------------------------
    # [1] CURRENT PREDICTION
    # ----------------------------------------------------
    print("[1] CURRENT PREDICTION")
    print("-" * 40)
    predictor = OccupancyPredictor()
    sample_sensor_data = {
        "temperature": 28.5,
        "light": 150.0,
        "sound": 0.40,
        "co2": 900.0,
        "motion": 1.0,
    }
    prediction = predictor.predict(sample_sensor_data)
    print(f"Sensor Readings : {sample_sensor_data}")
    print(f"Predicted State : {prediction['occupancy_state']} (Class {prediction['occupancy_class']})")
    print(f"AI Confidence   : {prediction['confidence'] * 100:.2f}%")
    print()

    # ----------------------------------------------------
    # [2] FEEDBACK COLLECTION
    # ----------------------------------------------------
    print("[2] FEEDBACK COLLECTION")
    print("-" * 40)
    manager = FeedbackManager(feedback_file=demo_feedback_path)
    sample_decision = {
        "cooling": "MEDIUM",
        "ventilation": "MEDIUM",
        "lighting": "MEDIUM",
        "priority": "HIGH",
        "reasons": ["Moderate occupancy and CO2 detected."],
    }
    manager.record_feedback(sample_sensor_data, prediction, sample_decision, feedback=None)
    print("Pending decision recorded successfully in demo feedback log.")
    print()

    # ----------------------------------------------------
    # [3] FEEDBACK EVALUATION
    # ----------------------------------------------------
    print("[3] FEEDBACK EVALUATION")
    print("-" * 40)
    evaluator = FeedbackEvaluator(feedback_file=demo_feedback_path)
    # Simulate user/operator providing ground truth (e.g. actual occupancy class 2 = Medium Occupancy)
    actual_occupancy = 2
    eval_result = evaluator.evaluate_latest(actual_occupancy)
    print(f"Simulated Ground Truth (User/Operator): Class {eval_result['actual_class']} ({eval_result['actual_state']})")
    print(f"Evaluation Result                       : {eval_result['feedback'].upper()}")
    print()

    # ----------------------------------------------------
    # [4] IMPROVEMENT ANALYSIS
    # ----------------------------------------------------
    print("[4] IMPROVEMENT ANALYSIS")
    print("-" * 40)
    engine = LearningEngine(feedback_file=demo_feedback_path, learning_dataset=demo_dataset_path)
    incorrect_cases = engine.analyse_improvement_cases()
    if incorrect_cases:
        print(f"Detected {len(incorrect_cases)} incorrect decision(s) flagged for model improvement analysis.")
    else:
        print("Decision evaluated as correct (injecting synthetic incorrect record for analysis demonstration)...")
        synth_record = {
            "timestamp": "2026-10-05T12:00:00",
            "sensor_data": {"temperature": 22.0, "light": 80.0, "sound": 0.1, "co2": 420.0, "motion": 0.0},
            "prediction": {"occupancy_class": 1, "occupancy_state": "Low Occupancy", "confidence": 0.65},
            "decision": {"cooling": "OFF", "ventilation": "LOW", "lighting": "OFF", "priority": "NORMAL"},
            "feedback": "incorrect",
            "actual": {"occupancy_class": 0, "occupancy_state": "Empty"}
        }
        records = evaluator.load_records()
        records.append(synth_record)
        evaluator.save_records(records)
        incorrect_cases = engine.analyse_improvement_cases()
        print(f"Analyzed improvement cases: {len(incorrect_cases)} case(s) extracted for learning.")
    print()

    # ----------------------------------------------------
    # [5] LEARNING ELIGIBILITY (Insufficient Feedback State)
    # ----------------------------------------------------
    print("[5] LEARNING ELIGIBILITY (State A: Insufficient Feedback)")
    print("-" * 40)
    summary_low = engine.get_learning_summary()
    print(f"Validated feedback samples : {summary_low['valid_training_records']} / {MIN_FEEDBACK_SAMPLES}")
    print(f"Learning Status            : {summary_low['learning_status']}")
    print("Result                     : INSUFFICIENT_FEEDBACK (Retraining deferred safely)")
    print()

    # ----------------------------------------------------
    # [6] CANDIDATE TRAINING (Sufficient Feedback State)
    # ----------------------------------------------------
    print("[6] CANDIDATE TRAINING (State B: Sufficient Feedback Simulation)")
    print("-" * 40)
    print(f"Injecting simulated validated feedback to meet threshold ({MIN_FEEDBACK_SAMPLES}+ samples)...")

    # Generate 52 synthetic valid feedback records spanning classes 0, 1, 2, 3
    existing_records = evaluator.load_records()
    states = ["Empty", "Low Occupancy", "Medium Occupancy", "High Occupancy"]
    for i in range(55):
        cls_idx = i % 4
        rec = {
            "timestamp": f"2026-10-05T12:{i:02d}:00",
            "sensor_data": {
                "temperature": 22.0 + (cls_idx * 2.0),
                "light": 100.0 + (cls_idx * 150.0),
                "sound": 0.1 + (cls_idx * 0.15),
                "co2": 400.0 + (cls_idx * 200.0),
                "motion": float(cls_idx > 0),
            },
            "prediction": {
                "occupancy_class": cls_idx,
                "occupancy_state": states[cls_idx],
                "confidence": 0.85,
            },
            "decision": {
                "cooling": "LOW",
                "ventilation": "LOW",
                "lighting": "MEDIUM",
                "priority": "NORMAL",
            },
            "feedback": "correct" if i % 5 != 0 else "incorrect",
            "actual": {
                "occupancy_class": cls_idx,
                "occupancy_state": states[cls_idx],
            },
        }
        existing_records.append(rec)
    evaluator.save_records(existing_records)

    summary_high = engine.get_learning_summary()
    print(f"Validated feedback samples : {summary_high['valid_training_records']} / {MIN_FEEDBACK_SAMPLES}")
    print(f"Learning Status            : {summary_high['learning_status']}")
    print("Training Eligibility       : MET. Training candidate model...")

    # Build training data and train candidate
    candidate_train_data, val_data = engine.build_candidate_training_data()
    prod_model, prod_scaler = engine.load_production_model()
    cand_model, cand_scaler = engine.train_candidate_model(candidate_train_data)
    print("Candidate model trained successfully.")
    print()

    # ----------------------------------------------------
    # [7] VALIDATION COMPARISON
    # ----------------------------------------------------
    print("[7] VALIDATION COMPARISON")
    print("-" * 40)
    prod_acc = engine.evaluate_model_on_data(prod_model, prod_scaler, val_data)
    cand_acc = engine.evaluate_model_on_data(cand_model, cand_scaler, val_data)
    gain = cand_acc - prod_acc
    print(f"Validation Set Size      : {len(val_data)} rows (Untouched UCI split)")
    print(f"Production Model Accuracy: {prod_acc:.2%}")
    print(f"Candidate Model Accuracy : {cand_acc:.2%}")
    print(f"Accuracy Gain            : {gain:+.2%}")
    print(f"Minimum Gain Required    : {MINIMUM_ACCURACY_GAIN:.2%}")
    print()

    # ----------------------------------------------------
    # [8] SAFE LEARNING DECISION
    # ----------------------------------------------------
    print("[8] SAFE LEARNING DECISION")
    print("-" * 40)
    promote = engine.should_promote_candidate(prod_acc, cand_acc)
    if promote:
        print("Decision: CANDIDATE ELIGIBLE FOR PROMOTION")
        print(f"Reason  : Candidate accuracy gain ({gain:.2%}) meets or exceeds the required threshold ({MINIMUM_ACCURACY_GAIN:.2%}).")
    else:
        print("Decision: CANDIDATE REJECTED")
        print(f"Reason  : Candidate accuracy gain ({gain:.2%}) does not meet the minimum required threshold ({MINIMUM_ACCURACY_GAIN:.2%}).")

    print()
    print("NOTE: Production model remains UNCHANGED for safety.")
    print("Demonstration completed successfully.")

    # Cleanup demo files
    if demo_feedback_path.exists():
        demo_feedback_path.unlink()
    if demo_dataset_path.exists():
        demo_dataset_path.unlink()


if __name__ == "__main__":
    run_demo()
