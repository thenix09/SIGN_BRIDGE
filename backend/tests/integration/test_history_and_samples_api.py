"""Integration tests for History and Sample collection API endpoints."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from backend.app.domain.translation import Translation
from backend.app.main import app
from backend.app.repositories.sample_repo import SQLiteSampleRepository
from backend.app.repositories.translation_repo import SQLiteTranslationRepository
from backend.app.services.history import HistoryService

client = TestClient(app)


def _sample_hand_landmarks() -> list[list[float]]:
    """Return 21 dummy 3D landmarks."""
    return [[float(i) * 0.01, float(i) * 0.02, 0.0] for i in range(21)]


def test_history_endpoints_lifecycle() -> None:
    # 1. Clear history first
    del_resp = client.delete("/api/history")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"ok": True}

    # 2. History should be empty
    get_resp = client.get("/api/history?limit=10")
    assert get_resp.status_code == 200
    assert get_resp.json() == {"items": []}

    # 3. Add a translation through repository
    repo = SQLiteTranslationRepository()
    service = HistoryService(repo=repo)
    service.record(
        Translation(
            id=None,
            direction="sign_to_text",
            language="ASL",
            content="HELLO WORLD",
            created_at=datetime.now(UTC),
        )
    )

    # 4. History should now contain the recorded item
    get_resp2 = client.get("/api/history?limit=10")
    assert get_resp2.status_code == 200
    items = get_resp2.json()["items"]
    assert len(items) == 1
    assert items[0]["content"] == "HELLO WORLD"
    assert items[0]["direction"] == "sign_to_text"
    assert items[0]["language"] == "ASL"

    # 5. Clear again
    del_resp2 = client.delete("/api/history")
    assert del_resp2.status_code == 200
    assert client.get("/api/history").json() == {"items": []}


def test_samples_endpoints_lifecycle() -> None:
    # Clear sample repo before test
    sample_repo = SQLiteSampleRepository()
    sample_repo.clear()

    # 1. Create a valid sample for label 'A'
    payload = {
        "label": "A",
        "handedness": "Right",
        "landmarks": _sample_hand_landmarks(),
        "group": "test-group",
    }
    create_resp = client.post("/api/samples", json=payload)
    assert create_resp.status_code == 200
    data = create_resp.json()
    assert "id" in data
    assert data["count_for_label"] >= 1

    # 2. Create another sample for label 'A' and one for 'B'
    client.post("/api/samples", json=payload)
    payload_b = {
        "label": "B",
        "handedness": "Left",
        "landmarks": _sample_hand_landmarks(),
        "group": "test-group",
    }
    client.post("/api/samples", json=payload_b)

    # 3. Get stats
    stats_resp = client.get("/api/samples/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total"] >= 3
    assert stats["counts"]["A"] >= 2
    assert stats["counts"]["B"] >= 1

    # 4. Export CSV
    export_resp = client.get("/api/samples/export")
    assert export_resp.status_code == 200
    assert "text/csv" in export_resp.headers["content-type"]
    csv_text = export_resp.text
    lines = csv_text.strip().splitlines()
    assert len(lines) >= 4  # Header + 3 samples
    header = lines[0].split(",")
    assert header[0] == "label"
    assert header[1] == "group"
    assert header[2] == "f0"
    assert header[-1] == "f62"


def test_samples_validation_errors() -> None:
    # Invalid label (multiple characters)
    bad_label = {
        "label": "ABC",
        "handedness": "Right",
        "landmarks": _sample_hand_landmarks(),
    }
    resp1 = client.post("/api/samples", json=bad_label)
    assert resp1.status_code == 422

    # Invalid landmarks count (not 21)
    bad_landmarks = {
        "label": "A",
        "handedness": "Right",
        "landmarks": [[0.1, 0.2, 0.3]],  # only 1 landmark
    }
    resp2 = client.post("/api/samples", json=bad_landmarks)
    assert resp2.status_code == 422
