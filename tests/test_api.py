import pytest
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_health():
	response = client.get("/health")
	assert response.status_code == 200
	assert response.json()["status"] == "ok"


def test_prediction():
	# Synthetic features test the API contract, not model accuracy.
	# No dataset download is needed to run these tests.
	features = [0.0] * 54 + [1.0, 1.0, 1.0]

	response = client.post("/predict", json={"features": features})

	assert response.status_code == 200
	result = response.json()
	assert result["prediction"] in [0, 1]
	assert len(result["probabilities"]) == 2
	assert all(0.0 <= p <= 1.0 for p in result["probabilities"])
	assert sum(result["probabilities"]) == pytest.approx(1.0)


@pytest.mark.parametrize("feature_count", [0, 2, 30, 56, 58])
def test_invalid_feature_count(feature_count):
	response = client.post("/predict", json={"features": [0.0] * feature_count})
	assert response.status_code == 422


def test_non_numeric_feature():
	response = client.post("/predict", json={"features": ["not-a-number"] + [0.0] * 56})
	assert response.status_code == 422


def test_missing_features():
	response = client.post("/predict", json={})
	assert response.status_code == 422