document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const badgeMqtt = document.getElementById("badge-mqtt");
    const badgeEsp32 = document.getElementById("badge-esp32");
    const badgeScenario = document.getElementById("badge-scenario");
    const badgeWs = document.getElementById("badge-ws");

    const scenarioContainer = document.getElementById("scenario-buttons-container");
    const intervalButtons = document.querySelectorAll(".btn-interval");

    // Sensor Value & Bar Elements
    const valTemp = document.getElementById("val-temp");
    const barTemp = document.getElementById("bar-temp");

    const valHumidity = document.getElementById("val-humidity");
    const barHumidity = document.getElementById("bar-humidity");

    const valLight = document.getElementById("val-light");
    const barLight = document.getElementById("bar-light");

    const valSound = document.getElementById("val-sound");
    const barSound = document.getElementById("bar-sound");

    const valCo2 = document.getElementById("val-co2");
    const barCo2 = document.getElementById("bar-co2");

    const valMotion = document.getElementById("val-motion");
    const indMotion = document.getElementById("ind-motion");
    const motionText = document.getElementById("motion-status-text");

    // Cognitive Elements
    const cogState = document.getElementById("cog-state");
    const cogConfidence = document.getElementById("cog-confidence");
    const cogClass = document.getElementById("cog-class");

    // Decision Elements
    const decPriority = document.getElementById("dec-priority");
    const decReasons = document.getElementById("dec-reasons");

    // Actuator Elements
    const actCooling = document.getElementById("act-cooling");
    const actVentilation = document.getElementById("act-ventilation");
    const actLighting = document.getElementById("act-lighting");

    const errorToast = document.getElementById("error-toast");

    let currentScenario = "random";
    let availableScenarios = [];

    function showError(message) {
        errorToast.textContent = message;
        errorToast.classList.remove("hidden");
        setTimeout(() => {
            errorToast.classList.add("hidden");
        }, 4000);
    }

    // Fetch available scenarios on load
    async function fetchScenarios() {
        try {
            const res = await fetch("/api/scenarios");
            if (res.ok) {
                availableScenarios = await res.json();
                renderScenarioButtons();
            }
        } catch (e) {
            console.error("Failed to fetch scenarios:", e);
        }
    }

    function renderScenarioButtons() {
        scenarioContainer.innerHTML = "";
        availableScenarios.forEach(scn => {
            const btn = document.createElement("button");
            btn.className = `btn-scenario ${scn === currentScenario ? "active" : ""}`;
            btn.textContent = scn.replace("_", " ");
            btn.dataset.scenario = scn;
            btn.addEventListener("click", () => setScenario(scn));
            scenarioContainer.appendChild(btn);
        });
    }

    async function setScenario(scn) {
        try {
            const res = await fetch(`/api/scenario/${scn}`, { method: "POST" });
            if (res.ok) {
                currentScenario = scn;
                document.querySelectorAll(".btn-scenario").forEach(b => {
                    b.classList.toggle("active", b.dataset.scenario === scn);
                });
                badgeScenario.textContent = `Scenario: ${scn}`;
            } else {
                const err = await res.json();
                showError(err.detail || "Failed to set scenario");
            }
        } catch (e) {
            showError("Network error setting scenario");
        }
    }

    // Publish Interval Controls
    intervalButtons.forEach(btn => {
        btn.addEventListener("click", async () => {
            const interval = parseFloat(btn.dataset.interval);
            try {
                const res = await fetch("/api/publish-interval", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ interval: interval })
                });
                if (res.ok) {
                    intervalButtons.forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                } else {
                    const err = await res.json();
                    showError(err.detail || "Failed to set interval");
                }
            } catch (e) {
                showError("Network error setting interval");
            }
        });
    });

    // Update UI with System Data
    function updateDashboard(data) {
        if (!data) return;

        // Status
        const status = data.status || {};
        const bridge = status.bridge || {};
        const esp32 = status.esp32 || {};

        if (bridge.connected) {
            badgeMqtt.textContent = "MQTT: Connected";
            badgeMqtt.className = "badge connected";
        } else {
            badgeMqtt.textContent = "MQTT: Disconnected";
            badgeMqtt.className = "badge disconnected";
        }

        if (esp32.running) {
            badgeEsp32.textContent = "ESP32: Running";
            badgeEsp32.className = "badge running";
        } else {
            badgeEsp32.textContent = "ESP32: Stopped";
            badgeEsp32.className = "badge stopped";
        }

        if (esp32.scenario && esp32.scenario !== currentScenario) {
            currentScenario = esp32.scenario;
            document.querySelectorAll(".btn-scenario").forEach(b => {
                b.classList.toggle("active", b.dataset.scenario === currentScenario);
            });
            badgeScenario.textContent = `Scenario: ${currentScenario}`;
        }

        // Sensors
        const sensors = data.sensors || {};
        const temp = sensors.temperature !== undefined ? sensors.temperature : 0;
        const hum = sensors.humidity !== undefined ? sensors.humidity : 0;
        const light = sensors.light !== undefined ? sensors.light : 0;
        const sound = sensors.sound !== undefined ? sensors.sound : 0;
        const co2 = sensors.co2 !== undefined ? sensors.co2 : 400;
        const motion = sensors.motion !== undefined ? sensors.motion : 0;

        valTemp.textContent = `${temp.toFixed(1)} °C`;
        barTemp.style.width = `${Math.min(Math.max((temp - 15) / 25 * 100, 0), 100)}%`;

        valHumidity.textContent = `${hum.toFixed(1)} %`;
        barHumidity.style.width = `${Math.min(Math.max(hum, 0), 100)}%`;

        valLight.textContent = `${light.toFixed(1)} lux`;
        barLight.style.width = `${Math.min(Math.max(light / 1000 * 100, 0), 100)}%`;

        valSound.textContent = `${sound.toFixed(2)}`;
        barSound.style.width = `${Math.min(Math.max(sound * 100, 0), 100)}%`;

        valCo2.textContent = `${co2.toFixed(0)} ppm`;
        barCo2.style.width = `${Math.min(Math.max((co2 - 400) / 1100 * 100, 0), 100)}%`;

        if (motion > 0.5) {
            valMotion.textContent = "Detected (1)";
            indMotion.className = "motion-indicator motion-active";
            motionText.textContent = "Motion Present";
        } else {
            valMotion.textContent = "Clear (0)";
            indMotion.className = "motion-indicator";
            motionText.textContent = "No Motion";
        }

        // Cognitive
        const cognitive = data.cognitive || {};
        const prediction = cognitive.prediction || {};
        const decision = cognitive.decision || {};

        cogState.textContent = prediction.occupancy_state || "Waiting...";
        const conf = prediction.confidence !== undefined ? prediction.confidence * 100 : 0;
        cogConfidence.textContent = `${conf.toFixed(1)}%`;
        cogClass.textContent = prediction.occupancy_class !== undefined ? `Class ${prediction.occupancy_class}` : "--";

        // Decision
        const priority = decision.priority || "NORMAL";
        decPriority.textContent = priority;
        decPriority.className = `priority-badge ${priority.toLowerCase()}`;

        const reasons = decision.reasons || ["Waiting for inference..."];
        decReasons.innerHTML = "";
        reasons.forEach(r => {
            const li = document.createElement("li");
            li.textContent = r;
            decReasons.appendChild(li);
        });

        // Actuators
        const actuators = data.actuators || {};
        const esp32State = actuators.esp32_state || {};

        updateActuatorBadge(actCooling, esp32State.cooling || "OFF");
        updateActuatorBadge(actVentilation, esp32State.ventilation || "OFF");
        updateActuatorBadge(actLighting, esp32State.lighting || "OFF");
    }

    function updateActuatorBadge(element, state) {
        const val = state.toLowerCase();
        element.textContent = state;
        element.className = `actuator-badge ${val}`;
    }

    // REST Fallback Initial Load
    async function loadInitialState() {
        try {
            const res = await fetch("/api/system");
            if (res.ok) {
                const data = await res.json();
                updateDashboard(data);
            }
        } catch (e) {
            console.error("Initial REST fetch failed:", e);
        }
    }

    // WebSocket Connection Management
    let ws = null;
    let wsRetryCount = 0;

    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        badgeWs.textContent = "Connecting WS...";
        badgeWs.style.backgroundColor = "rgba(234, 179, 8, 0.2)";
        badgeWs.style.color = "#facc15";

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            badgeWs.textContent = "WebSocket Live";
            badgeWs.className = "badge connected";
            wsRetryCount = 0;
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            } catch (e) {
                console.error("Failed to parse WS message:", e);
            }
        };

        ws.onclose = () => {
            badgeWs.textContent = "Disconnected";
            badgeWs.className = "badge disconnected";
            
            // Reconnect with exponential backoff up to 10s
            wsRetryCount++;
            const timeout = Math.min(1000 * Math.pow(1.5, wsRetryCount), 10000);
            setTimeout(connectWebSocket, timeout);
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            ws.close();
        };
    }

    // Initialize
    fetchScenarios();
    loadInitialState();
    connectWebSocket();
});
