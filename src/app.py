from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

app = FastAPI(
	title="Spambase Spam Classification API",
	description="Classifies precomputed email features: 0 = not spam, 1 = spam.",
)
model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)


class PredictionRequest(BaseModel):
	features: list[float] = Field(
		min_length=57,
		max_length=57,
		description="57 numeric Spambase features in the original dataset order (feature_0 through feature_56), not raw email text.",
	)


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
	X = pd.DataFrame(
		[request.features], columns=feature_names
	)
	prediction = int(model.predict(X)[0])
	probability = model.predict_proba(X)[0].tolist()

	return {
		"prediction": prediction,
		"probabilities": probability,
	}