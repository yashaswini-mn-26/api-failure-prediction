import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

FEATURES = ["response_time", "status_code", "cpu_usage", "memory_usage"]
TARGET   = "failure"

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"Loaded {len(df):,} rows | Failure rate: {df[TARGET].mean()*100:.1f}%")
    print(df[FEATURES + [TARGET]].describe().round(2).to_string())
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train(data_path: str, model_path: str):
    df = load_and_validate(data_path)

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Cross-validation
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\nCV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n── Test Set Metrics ─────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Stable", "Failure"]))
    print(f"ROC-AUC:         {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Avg Precision:   {average_precision_score(y_test, y_proba):.4f}")

    # Feature importances
    clf = pipeline.named_steps["clf"]
    importances = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])
    print("\n── Feature Importances ──────────────────────────")
    for name, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {name:<20} {imp:.4f}  {bar}")

    # Save
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved → {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SentinelAPI failure predictor")
    parser.add_argument("--data", default="api_logs.csv", help="Training CSV path")
    parser.add_argument("--out",  default="model.pkl",    help="Output model path")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()