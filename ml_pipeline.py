# ml_pipeline.py

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data():
    data = load_iris()
    X = data["data"]
    y = data["target"]
    feature_names = data["feature_names"]
    target_names = data["target_names"]
    return X, y, feature_names, target_names

def train_and_evaluate():
    X, y, feature_names, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Model accuracy:", acc)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save model + metadata
    joblib.dump(
        {
            "model": pipeline,
            "feature_names": feature_names,
            "target_names": target_names
        },
        MODEL_PATH
    )
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
