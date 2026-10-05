# Cognitive IoT Framework for Context-Aware Autonomous Decision Making

<div align="center">

![Cognitive IoT](https://img.shields.io/badge/Cognitive-IoT-0f172a?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MQTT](https://img.shields.io/badge/MQTT-IoT%20Communication-660066?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**A modular Cognitive IoT prototype that senses its environment, understands context, makes autonomous decisions, controls actuators, and supports feedback-driven adaptation.**

### Sense → Understand → Decide → Act → Learn

</div>

---

## Project Overview

Traditional IoT systems generally follow a predefined pattern:

> **Sensor → Rule → Actuator**

This project proposes a **Cognitive IoT Framework** that introduces contextual understanding and AI-based decision making into the IoT loop.

Instead of simply reacting to individual sensor values, the framework combines multiple environmental signals, derives context, predicts occupancy state using machine learning, makes autonomous decisions, and communicates those decisions to virtual actuators.

The current prototype uses a **Virtual ESP32** and simulated sensors so that the complete cognitive-IoT pipeline can be demonstrated without physical hardware. The architecture is designed so that the Virtual ESP32 can later be replaced by a physical ESP32 and real sensors/actuators.

---

## Core Idea

```text
┌──────────────────┐
│  Environmental   │
│     Sensors      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Virtual ESP32   │
│  Sensor Gateway  │
└────────┬─────────┘
         │ MQTT
         ▼
┌──────────────────┐
│ Sensor Processing│
│  & Context       │
│    Fusion        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   AI / Cognitive │
│     Engine       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Decision Engine  │
│ Cooling / Light  │
│ Ventilation      │
└────────┬─────────┘
         │ MQTT
         ▼
┌──────────────────┐
│ Virtual Actuators│
└────────┬─────────┘
         │
         ▼
     Feedback
         │
         └──────────────► Learning
```

---

## Key Features

| Feature | Description |
|---|---|
| Cognitive Processing | Converts sensor observations into contextual decisions |
| Sensor Fusion | Combines temperature, lighting, sound, CO₂ and motion information |
| ML Occupancy Inference | Random Forest model predicts occupancy state |
| Autonomous Decisions | Determines cooling, ventilation and lighting levels |
| MQTT Communication | Connects the simulated IoT device and cognitive system |
| Virtual ESP32 | Software representation of the future physical IoT node |
| Web Dashboard | Live monitoring through HTML/CSS/JavaScript |
| FastAPI | REST APIs and WebSocket communication |
| Feedback Loop | Records and evaluates feedback for adaptive learning |
| Controlled Learning | Candidate models are promoted only when improvement meets a defined threshold |
| Modular Architecture | AI, decision, MQTT, simulation and API layers remain independently reusable |

---

# System Architecture

```text
                         ┌─────────────────────────┐
                         │       WEB BROWSER       │
                         │  HTML + CSS + JavaScript│
                         └────────────┬────────────┘
                                      │
                               HTTP / WebSocket
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │         FastAPI         │
                         │      Application API    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │    MQTT Broker   │
                            │   localhost:1883 │
                            └────────┬─────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   │                                   │
                   ▼                                   ▼
          ┌──────────────────┐                ┌──────────────────┐
          │   Virtual ESP32  │                │ Cognitive Bridge │
          │                  │── MQTT ───────►│                  │
          │ Sensors          │                │ CognitivePipeline│
          │ Actuators        │◄── MQTT ──────│ AI + Decisions   │
          └──────────────────┘                └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │  Occupancy Model │
                                               │  Random Forest   │
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │  Decision Engine │
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │ Virtual Actuator │
                                               │      System      │
                                               └──────────────────┘
```

---

# Cognitive Processing Cycle

The framework follows five conceptual stages:

```text
      ┌────────┐
      │ SENSE  │
      └───┬────┘
          ▼
   Sensor observations
          │
          ▼
    ┌────────────┐
    │ UNDERSTAND │
    └─────┬──────┘
          ▼
 Context + AI inference
          │
          ▼
      ┌────────┐
      │ DECIDE │
      └───┬────┘
          ▼
 Autonomous action selection
          │
          ▼
       ┌──────┐
       │ ACT  │
       └──┬───┘
          ▼
    Virtual actuators
          │
          ▼
      ┌────────┐
      │ LEARN  │
      └───┬────┘
          │
          └────────► Feedback / adaptation
```

### 1. Sense

The system receives environmental observations such as:

- Temperature
- Humidity
- Light
- Sound
- CO₂
- Motion/PIR activity

### 2. Understand

Raw readings are transformed into contextual features and passed to the ML model.

The current model predicts:

| Class | Occupancy State |
|---:|---|
| 0 | Empty |
| 1 | Low Occupancy |
| 2 | Medium Occupancy |
| 3 | High Occupancy |

### 3. Decide

The decision engine combines the inferred occupancy state with environmental thresholds.

Example:

```text
High Occupancy
      +
High Temperature
      +
Elevated CO₂
      │
      ▼
Cooling      → HIGH
Ventilation  → MEDIUM/HIGH
Lighting     → Based on light level
Priority     → HIGH/CRITICAL
```

### 4. Act

The selected actions are published through MQTT and applied to the Virtual ESP32's virtual actuator state.

### 5. Learn

Feedback can be recorded and evaluated. A candidate model is tested against an untouched validation set before it can replace the production model.

---

# Machine Learning

## Model

The current occupancy inference model uses a **Random Forest Classifier**.

```text
Raw sensor features
        │
        ▼
Leakage-safe scaling
        │
        ▼
Context feature generation
        │
        ▼
Random Forest
        │
        ▼
Occupancy class + confidence
```

### Input Features

The production model uses:

- `temperature_mean`
- `light_mean`
- `sound_mean`
- `co2`
- `pir_activity`

These are transformed into contextual features including:

- Temperature context
- Lighting context
- Sound context
- CO₂ context
- Motion context
- Unified context score

---

# Model Results

The project evaluates the model in more than one way because a random train/test split can produce overly optimistic results for time-dependent sensor data.

### Random Stratified Holdout

**Accuracy: 99.61%**

This is the standard 80/20 stratified evaluation used for the current model.

### Temporal Evaluation

**Approximately 91.1%**

A temporal evaluation is more representative of deployment conditions because it evaluates the model on later observations rather than randomly mixing observations from the same period.

> **Important:** The 99.61% result should not be interpreted as guaranteed real-world accuracy.

---

# Dataset

The current prototype primarily uses the **UCI Room Occupancy Estimation** dataset.

```text
Rows:       10,129
Target:     Room_Occupancy_Count
Classes:    0, 1, 2, 3
```

The dataset contains measurements involving:

- Temperature
- Light
- Sound
- CO₂
- PIR sensors
- Occupancy count

A second UCI occupancy dataset is also retained for future external validation/generalization experiments.

### Data Pipeline

```text
UCI Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Sensor Context / Fusion
    │
    ▼
Model Training
    │
    ▼
Evaluation
```

---

# MQTT Architecture

MQTT acts as the communication layer between the simulated IoT device and the cognitive system.

### MQTT Topics

| Topic | Purpose |
|---|---|
| `cognitive-iot/sensors` | Sensor observations |
| `cognitive-iot/actuators` | Cognitive actuator commands |
| `cognitive-iot/device/status` | Virtual device status |

### Message Flow

```text
Virtual ESP32
     │
     │ publish
     ▼
cognitive-iot/sensors
     │
     ▼
Cognitive Bridge
     │
     ▼
Cognitive Pipeline
     │
     ▼
cognitive-iot/actuators
     │
     ▼
Virtual ESP32
```

---

# Demonstration Scenarios

The Virtual ESP32 supports predefined scenarios:

```text
┌──────────────┐
│ Random       │
├──────────────┤
│ Empty        │
├──────────────┤
│ High         │
│ Occupancy    │
├──────────────┤
│ Critical CO₂ │
└──────────────┘
```

These scenarios control simulated sensor inputs.

**Important:** the scenario name does not directly determine the AI prediction. The ML model remains the authority for occupancy inference.

This allows the system to demonstrate how changing environmental conditions influence cognitive decisions without hard-coding the prediction.

---

# Feedback & Adaptive Learning

The framework contains a controlled learning mechanism.

```text
Feedback
   │
   ▼
Validation
   │
   ▼
Learning Dataset
   │
   ▼
Candidate Model
   │
   ▼
Untouched Validation
   │
   ├──── Improvement sufficient ────► Promote
   │
   └──── Improvement insufficient ──► Reject
```

The current promotion requirement is:

> **Candidate accuracy must improve by at least 0.50 percentage points.**

### Demonstration Result

| Model | Accuracy |
|---|---:|
| Production | 99.61% |
| Candidate | 99.75% |
| Improvement | +0.15 percentage points |
| Required | +0.50 percentage points |
| Result | **Rejected** |

The candidate improved accuracy by **0.15 percentage points**, from 99.61% to 99.75%, but this was below the required 0.50 percentage-point improvement threshold.

> Feedback labels used in the current prototype are simulated/user-provided ground truth. They are not measurements from physical occupancy sensors.

---

# Web Application

The final prototype uses a custom web interface instead of relying on Streamlit.

### Frontend

```text
HTML5
CSS3
Vanilla JavaScript
WebSocket
```

### Backend

```text
FastAPI
   │
   ├── REST API
   └── WebSocket
```

The dashboard provides live visibility into:

- MQTT connection
- Virtual ESP32 status
- Current scenario
- Sensor telemetry
- AI occupancy state
- AI confidence
- Decision priority
- Decision reasons
- Cooling state
- Ventilation state
- Lighting state
- Cognitive processing cycle

---

# Project Structure

```text
Cognitive-IoT/
│
├── actuators/
│   └── virtual_actuators.py
│
├── ai/
│   ├── analyze_temporal_shift.py
│   ├── evaluate.py
│   ├── evaluate_leakage.py
│   ├── evaluate_shift_impact.py
│   ├── evaluate_temporal.py
│   ├── predict.py
│   └── train_model.py
│
├── api/
│   ├── app.py
│   └── test_api.py
│
├── cognitive/
│   └── pipeline.py
│
├── config/
│   └── config.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets/
│
├── dataset/
│   ├── occupancy_detection/
│   └── room_occupancy_estimation/
│
├── decision/
│   └── decision_engine.py
│
├── feedback/
│   ├── feedback_evaluator.py
│   └── feedback_manager.py
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── fusion/
│   └── sensor_fusion.py
│
├── learning/
│   ├── demo_learning.py
│   └── learning_engine.py
│
├── mqtt/
│   ├── cognitive_bridge.py
│   ├── dashboard_client.py
│   ├── publisher.py
│   └── subscriber.py
│
├── preprocessing/
│   ├── data_cleaning.py
│   └── feature_engineering.py
│
├── research/
│
├── simulation/
│   ├── realtime_simulator.py
│   ├── scenarios.py
│   ├── sensor_simulator.py
│   └── virtual_esp32.py
│
├── tests/
├── utils/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Installation

## 1. Clone the Repository

```powershell
git clone https://github.com/Null35Pointer28/cognitive-iot-framework.git
cd cognitive-iot-framework
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 4. Start an MQTT Broker

The current development setup uses **Mosquitto** locally.

Default configuration:

```text
Host: localhost
Port: 1883
```

Verify that the broker is accessible before starting the application.

---

# Running the Project

## Complete Web Application

From the project root:

```powershell
python -m uvicorn api.app:app --reload
```

Open the dashboard:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Virtual ESP32

```powershell
python simulation\virtual_esp32.py
```

## Cognitive MQTT Bridge

```powershell
python mqtt\cognitive_bridge.py
```

## Original MQTT Pipeline

Sensor publisher:

```powershell
python mqtt\publisher.py
```

Subscriber:

```powershell
python -m mqtt.subscriber
```

Example scenarios:

```powershell
python mqtt\publisher.py --scenario empty
python mqtt\publisher.py --scenario high_occupancy
python mqtt\publisher.py --scenario critical_co2
```

---

# Testing

The project contains tests and validation scripts covering:

- Sensor simulation
- Sensor fusion
- AI prediction
- Decision logic
- API endpoints
- MQTT communication
- Virtual ESP32 integration
- Cognitive pipeline behavior
- Model leakage evaluation
- Temporal evaluation
- Distribution-shift analysis

FastAPI also provides an interactive Swagger interface:

```text
http://127.0.0.1:8000/docs
```

---

# Design Principles

## Separation of Concerns

```text
Simulation
    ↓
Communication
    ↓
Cognitive Processing
    ↓
Decision
    ↓
Actuation
    ↓
Feedback
```

Each layer has a distinct responsibility.

## Reusable Cognitive Core

The `CognitivePipeline` does not depend on MQTT or the web frontend.

This allows the same cognitive logic to be reused by:

- MQTT
- FastAPI
- Future physical ESP32 integration
- Future edge deployment

## No Duplicated AI Logic

The MQTT bridge and API do not implement separate prediction or decision algorithms.

They call the same cognitive pipeline.

---

# Future Hardware Integration

The current Virtual ESP32 is intentionally designed as a replacement point for physical hardware.

## Current Prototype

```text
Virtual Sensors
      ↓
Virtual ESP32
      ↓
MQTT
      ↓
Cognitive Engine
      ↓
Virtual Actuators
```

## Future Physical System

```text
DHT22 / Temperature Sensor
BH1750 / Light Sensor
PIR / Motion Sensor
CO₂ Sensor
        │
        ▼
      ESP32
        │
       MQTT
        │
        ▼
Cognitive IoT Engine
        │
       MQTT
        │
        ▼
Physical Actuators
Fan / Light / Ventilation
```

The cognitive engine and decision architecture can remain unchanged while the simulated hardware layer is replaced.

---

# Future Scope

Potential extensions include:

- Physical ESP32 deployment
- Real environmental sensors
- Physical actuator control
- Edge AI deployment
- Advanced sensor fusion
- External validation using additional datasets
- Improved adaptive learning
- Energy-aware decision making
- Predictive environmental control
- Multi-room Cognitive IoT
- Cloud-based monitoring
- Mobile application
- Security and authentication
- Distributed IoT deployment

---

# Current Limitations

1. Sensor readings are simulated rather than collected from physical hardware.
2. The Virtual ESP32 represents the hardware layer in software.
3. The occupancy model is trained primarily using UCI datasets.
4. Feedback labels in the adaptive-learning demonstration are simulated/user-provided.
5. The current decision engine uses defined environmental thresholds in addition to AI inference.
6. The prototype is intended as a research/PBL demonstration rather than a production building-control system.

---

# Team

**Project:** Cognitive IoT Framework for Context-Aware Autonomous Decision Making

**Program:** B.Tech CSE Internet of Things & Intelligent Systems

**Institution:** Manipal University Jaipur

---

<div align="center">

### Cognitive IoT

**Sense the environment. Understand the context. Decide intelligently. Act autonomously. Learn continuously.**

</div>
