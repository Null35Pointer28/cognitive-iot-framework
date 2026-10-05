"""
Cognitive IoT - FastAPI Application Layer

Exposes the cognitive architecture via REST endpoints and real-time WebSockets:
- System & component status (/api/status, /api/system)
- Live sensor telemetry (/api/sensors)
- Cognitive understanding & outputs (/api/cognitive)
- Virtual actuator commands & states (/api/actuators)
- Scenario controls & simulation intervals (/api/scenarios, /api/scenario/{name}, /api/publish-interval)
- Real-time state streaming (/ws)
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from simulation.virtual_esp32 import VirtualESP32
from mqtt.cognitive_bridge import CognitiveMQTTBridge
from simulation.scenarios import SCENARIOS


# Global backend instances for Virtual ESP32 and MQTT Bridge
esp32_instance: VirtualESP32 = None
bridge_instance: CognitiveMQTTBridge = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application Startup
    global esp32_instance, bridge_instance

    bridge_instance = CognitiveMQTTBridge(record_feedback=True)
    bridge_instance.start()

    esp32_instance = VirtualESP32(scenario="random", publish_interval_seconds=5)
    esp32_instance.start()

    yield

    # Application Shutdown
    if bridge_instance:
        bridge_instance.stop()
    if esp32_instance:
        esp32_instance.stop()


app = FastAPI(
    title="Cognitive IoT Framework API",
    description="Backend API for Context-Aware Autonomous Decision Making",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve frontend static files and root index.html
frontend_dir = root_dir / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def read_index():
        return FileResponse(str(frontend_dir / "index.html"))

# Enable CORS for local HTML/JS frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REST ENDPOINTS
# ============================================================

@app.get("/api/status")
def get_status():
    """Returns system and bridge operational status."""
    if bridge_instance is None or esp32_instance is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {
        "bridge": bridge_instance.get_status(),
        "esp32": esp32_instance.get_status(),
    }


@app.get("/api/sensors")
def get_sensors():
    """Returns the latest sensor telemetry readings."""
    if bridge_instance is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return bridge_instance.get_latest_sensor_data()


@app.get("/api/cognitive")
def get_cognitive():
    """Returns the latest AI prediction and cognitive decision output."""
    if bridge_instance is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return bridge_instance.get_latest_cognitive_result()


@app.get("/api/actuators")
def get_actuators():
    """Returns the latest published actuator command and virtual actuator state."""
    if bridge_instance is None or esp32_instance is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {
        "latest_command": bridge_instance.get_latest_actuator_command(),
        "esp32_state": esp32_instance.get_actuator_state(),
    }


@app.get("/api/system")
def get_system():
    """Returns a combined comprehensive JSON response of status, sensors, cognitive, and actuators."""
    if bridge_instance is None or esp32_instance is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {
        "status": {
            "bridge": bridge_instance.get_status(),
            "esp32": esp32_instance.get_status(),
        },
        "sensors": bridge_instance.get_latest_sensor_data(),
        "cognitive": bridge_instance.get_latest_cognitive_result(),
        "actuators": {
            "latest_command": bridge_instance.get_latest_actuator_command(),
            "esp32_state": esp32_instance.get_actuator_state(),
        },
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Returns the list of available simulation scenarios."""
    return list(SCENARIOS.keys())


@app.post("/api/scenario/{scenario_name}")
def set_scenario(scenario_name: str):
    """Changes the active Virtual ESP32 simulation scenario."""
    if esp32_instance is None:
        raise HTTPException(status_code=503, detail="Virtual ESP32 not running")
    if scenario_name not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario '{scenario_name}'. Available: {list(SCENARIOS.keys())}"
        )
    try:
        esp32_instance.set_scenario(scenario_name)
        return {"success": True, "active_scenario": scenario_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class IntervalRequest(BaseModel):
    interval: float = Field(..., gt=0, description="Publish interval in seconds")


@app.post("/api/publish-interval")
def set_publish_interval(req: IntervalRequest):
    """Sets the Virtual ESP32 sensor publishing interval in seconds."""
    if esp32_instance is None:
        raise HTTPException(status_code=503, detail="Virtual ESP32 not running")
    if req.interval < 0.5:
        raise HTTPException(status_code=400, detail="Publish interval must be at least 0.5 seconds.")
    try:
        esp32_instance.set_publish_interval(int(req.interval))
        return {"success": True, "publish_interval_seconds": req.interval}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# WEBSOCKET STREAMING
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Streams live system telemetry and cognitive state to connected frontend clients periodically."""
    await websocket.accept()
    try:
        while True:
            if bridge_instance is not None and esp32_instance is not None:
                state_payload = {
                    "status": {
                        "bridge": bridge_instance.get_status(),
                        "esp32": esp32_instance.get_status(),
                    },
                    "sensors": bridge_instance.get_latest_sensor_data(),
                    "cognitive": bridge_instance.get_latest_cognitive_result(),
                    "actuators": {
                        "latest_command": bridge_instance.get_latest_actuator_command(),
                        "esp32_state": esp32_instance.get_actuator_state(),
                    },
                }
                await websocket.send_json(state_payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
