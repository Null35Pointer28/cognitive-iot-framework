"""
Cognitive IoT - MQTT Monitoring Dashboard

Displays real-time sensor data, AI inference understanding,
cognitive decisions, and actuator responses received through the MQTT layer.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running via streamlit run dashboard/app.py
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from mqtt.dashboard_client import DashboardMQTTClient


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Cognitive IoT Framework",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Cognitive IoT Framework")
st.caption("Context-Aware Autonomous Decision Making")


# =========================================================
# MQTT CLIENT INITIALIZATION (Streamlit Lifecycle)
# =========================================================

@st.cache_resource
def get_cached_mqtt_client():
    client = DashboardMQTTClient()
    client.start()
    return client


mqtt_client = get_cached_mqtt_client()


# =========================================================
# 2. SYSTEM STATUS
# =========================================================

st.subheader("System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

is_connected = False
try:
    is_connected = mqtt_client.is_connected()
except Exception:
    is_connected = False

sensor_data = None
try:
    sensor_data = mqtt_client.get_sensor_data()
except Exception:
    sensor_data = None

actuator_data = None
try:
    actuator_data = mqtt_client.get_actuator_data()
except Exception:
    actuator_data = None

with status_col1:
    if is_connected:
        st.metric("MQTT Connection", "Connected")
    else:
        st.metric("MQTT Connection", "Disconnected")

with status_col2:
    if sensor_data is not None:
        st.metric("Inference Engine", "Active")
    else:
        st.metric("Inference Engine", "Waiting")

with status_col3:
    if actuator_data is not None:
        st.metric("Decision Layer", "Active")
    else:
        st.metric("Decision Layer", "Waiting")

with status_col4:
    if actuator_data is not None:
        st.metric("Actuator Interface", "Active")
    else:
        st.metric("Actuator Interface", "Waiting")


st.divider()


# =========================================================
# 3. PROCESSING ARCHITECTURE
# =========================================================

st.subheader("Processing Architecture")

arch_cols = st.columns(6)

pipeline_steps = [
    "Sensors",
    "MQTT",
    "AI / Sensor Fusion",
    "Cognitive Decision",
    "Actuators",
    "Feedback & Learning"
]

for idx, col in enumerate(arch_cols):
    with col:
        st.markdown(f"**{pipeline_steps[idx]}**")
        if idx < len(pipeline_steps) - 1:
            st.markdown("→")


st.divider()


# =========================================================
# 4. LIVE SENSOR DATA
# =========================================================

st.subheader("Live Sensor Data")

if sensor_data is None or not isinstance(sensor_data, dict):
    st.info("Waiting for sensor data...")
else:
    try:
        temp = float(sensor_data.get("temperature", 0.0))
        hum = float(sensor_data.get("humidity", 0.0))
        light = float(sensor_data.get("light", 0.0))
        sound = float(sensor_data.get("sound", 0.0))
        co2 = float(sensor_data.get("co2", 0.0))
        motion = float(sensor_data.get("motion", 0.0))

        s_col1, s_col2, s_col3, s_col4, s_col5, s_col6 = st.columns(6)

        with s_col1:
            st.metric("Temperature", f"{temp:.2f} °C")
        with s_col2:
            st.metric("Humidity", f"{hum:.2f} %")
        with s_col3:
            st.metric("Light", f"{light:.2f} lux")
        with s_col4:
            st.metric("Sound", f"{sound:.2f}")
        with s_col5:
            st.metric("CO₂", f"{co2:.2f} ppm")
        with s_col6:
            motion_state = "Detected (1)" if motion > 0.5 else "Clear (0)"
            st.metric("Motion", motion_state)
    except Exception as e:
        st.warning(f"Error parsing sensor data: {e}")


st.divider()


# =========================================================
# 5. COGNITIVE OUTPUT
# =========================================================

st.subheader("Cognitive Output")

if actuator_data is None or not isinstance(actuator_data, dict):
    st.info("Waiting for cognitive output...")
else:
    try:
        occupancy_state = actuator_data.get("occupancy_state", "Unknown")
        confidence = float(actuator_data.get("confidence", 0.0))
        priority = actuator_data.get("priority", "Unknown")

        c_col1, c_col2, c_col3 = st.columns(3)

        with c_col1:
            st.metric("Occupancy", occupancy_state)
        with c_col2:
            st.metric("AI Confidence", f"{confidence * 100:.2f}%")
        with c_col3:
            st.metric("Decision Priority", priority)
    except Exception as e:
        st.warning(f"Error parsing cognitive output: {e}")


st.divider()


# =========================================================
# 6. DECISION LAYER
# =========================================================

st.subheader("Decision Layer")

if actuator_data is None or not isinstance(actuator_data, dict):
    st.info("Waiting for actuator decision states...")
else:
    try:
        cooling = actuator_data.get("cooling", "OFF")
        ventilation = actuator_data.get("ventilation", "OFF")
        lighting = actuator_data.get("lighting", "OFF")

        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            st.metric("Cooling", cooling)
        with d_col2:
            st.metric("Ventilation", ventilation)
        with d_col3:
            st.metric("Lighting", lighting)
    except Exception as e:
        st.warning(f"Error parsing decision layer data: {e}")


st.divider()


# =========================================================
# 7. COMMUNICATION LAYER
# =========================================================

st.subheader("Communication Layer")

c_col1, c_col2, c_col3 = st.columns(3)

with c_col1:
    st.text_input("MQTT Broker", value="localhost:1883", disabled=True)

with c_col2:
    st.text_input("Sensor Data Topic", value="cognitive-iot/sensors", disabled=True)
    st.caption("Carries sensor readings toward the AI processing layer.")

with c_col3:
    st.text_input("Decision Data Topic", value="cognitive-iot/actuators", disabled=True)
    st.caption("Carries cognitive decisions toward the actuator/dashboard layer.")

st.markdown("")
st.markdown("**Data Flow Architecture:**")
flow_col1, flow_col2 = st.columns(2)
with flow_col1:
    st.markdown("`Sensor Data` → `MQTT Broker` → `AI Processing`")
with flow_col2:
    st.markdown("`AI Decision` → `MQTT Broker` → `Actuators / Dashboard`")


st.divider()


# =========================================================
# 8. REFRESH
# =========================================================

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("Refresh Data"):
        st.rerun()
