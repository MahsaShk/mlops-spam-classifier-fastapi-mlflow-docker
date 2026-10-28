import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

mlflow.set_tracking_uri("file://" + str(ROOT / "mlruns"))
mlflow.set_experiment("spambase-classifier")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
data = pd.read_csv(DATA_URL, header=None)
X = data.iloc[:, :-1].copy()
X.columns = [f"feature_{i}" for i in range(X.shape[1])]
y = data.iloc[:, -1].astype(int)  # 0 = not spam, 1 = spam

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=50, stratify=y
)

with mlflow.start_run(tags={"mlflow.user": os.environ.get("MLFLOW_USER", "mshk")}):
	model = RandomForestClassifier(
		n_estimators=100, random_state=42
	)
	model.fit(X_train, y_train)

	predictions = model.predict(X_test)
	accuracy = accuracy_score(y_test, predictions)

	mlflow.log_param("n_estimators", 100)
	mlflow.log_param("dataset", "uci-spambase")
	mlflow.log_metric("test_accuracy", float(accuracy))
	mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(1))

	joblib.dump(model, ARTIFACTS / "model.joblib")
	print(f"Spambase: {len(data)} emails, {X.shape[1]} features")
	print(f"Training samples: {len(X_train)}; test samples: {len(X_test)}")
	print(f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
	print(f"Saved model to {ARTIFACTS / 'model.joblib'}")