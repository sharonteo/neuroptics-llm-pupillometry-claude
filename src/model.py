import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_data(path="data/synthetic_pupillometry.csv"):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.drop(columns=["patient_id"])

    y = df["severity"]
    X = df.drop(columns=["severity"])

    # Encode target labels for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label encoder classes
    os.makedirs("artifacts", exist_ok=True)
    pd.Series(le.classes_).to_csv("artifacts/label_encoder_classes.csv", index=False)

    categorical = ["sex", "diagnosis", "site_id"]
    numeric = [col for col in X.columns if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric)
        ]
    )

    return X, y_encoded, preprocessor


def train_models(X_train, y_train, preprocessor):
    models = {
        "logistic_regression": LogisticRegression(max_iter=300),
        "random_forest": RandomForestClassifier(n_estimators=250),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss"
        )
    }

    trained = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        trained[name] = pipe

        # Save trained pipeline
        model_dir = f"artifacts/{name}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(pipe, f"{model_dir}/model.pkl")

    return trained


def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        report = classification_report(
            y_test, preds, output_dict=True
        )
        cm = confusion_matrix(y_test, preds)

        results[name] = {
            "classification_report": report,
            "confusion_matrix": cm
        }

    return results


def save_results(results, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    for name, metrics in results.items():
        model_dir = os.path.join(out_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        pd.DataFrame(metrics["classification_report"]).to_csv(
            os.path.join(model_dir, "classification_report.csv")
        )

        pd.DataFrame(metrics["confusion_matrix"]).to_csv(
            os.path.join(model_dir, "confusion_matrix.csv")
        )


def main():
    df = load_data()
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = train_models(X_train, y_train, preprocessor)
    results = evaluate_models(models, X_test, y_test)
    save_results(results)

    print("Model training complete. Results saved to artifacts/")


if __name__ == "__main__":
    main()