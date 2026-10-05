"""
Cognitive IoT - FastAPI Test Script

Tests all API endpoints, scenario switching, publishing interval updates,
static frontend serving, and WebSocket state streaming using FastAPI TestClient.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from api.app import app


def test_api():
    print("========================================")
    print("       FASTAPI BACKEND TEST SCRIPT")
    print("========================================")

    with TestClient(app) as client:
        # 0. Test Root / serving frontend index.html
        response = client.get("/")
        print(f"\nGET / -> Status: {response.status_code} (Content-Type: {response.headers.get('content-type')})")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

        # 1. Test /api/status
        response = client.get("/api/status")
        print(f"\nGET /api/status -> Status: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

        # 2. Test /api/scenarios
        response = client.get("/api/scenarios")
        print(f"\nGET /api/scenarios -> Status: {response.status_code}")
        scenarios = response.json()
        print(scenarios)
        assert response.status_code == 200
        assert "high_occupancy" in scenarios

        # 3. Test POST /api/scenario/{scenario_name}
        response = client.post("/api/scenario/high_occupancy")
        print(f"\nPOST /api/scenario/high_occupancy -> Status: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

        response = client.post("/api/scenario/critical_co2")
        print(f"\nPOST /api/scenario/critical_co2 -> Status: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

        # Test invalid scenario
        response = client.post("/api/scenario/nonexistent_scenario")
        print(f"\nPOST /api/scenario/nonexistent_scenario -> Status: {response.status_code}")
        assert response.status_code == 400

        # 4. Test POST /api/publish-interval
        response = client.post("/api/publish-interval", json={"interval": 2.0})
        print(f"\nPOST /api/publish-interval -> Status: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

        # Test invalid interval
        response = client.post("/api/publish-interval", json={"interval": 0.1})
        print(f"\nPOST /api/publish-interval (invalid) -> Status: {response.status_code}")
        assert response.status_code == 400

        # 5. Test data endpoints
        for endpoint in ["/api/sensors", "/api/cognitive", "/api/actuators", "/api/system"]:
            response = client.get(endpoint)
            print(f"\nGET {endpoint} -> Status: {response.status_code}")
            print(response.json())
            assert response.status_code == 200

        # 6. Test WebSocket /ws
        print("\nTesting WebSocket /ws...")
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            print("WebSocket received initial payload:")
            print(data)
            assert "status" in data
            assert "sensors" in data
            assert "cognitive" in data
            assert "actuators" in data

    print("\nAll FastAPI and Frontend-serving tests passed successfully.")


if __name__ == "__main__":
    test_api()
