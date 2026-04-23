"""
Mission Intent Classifier Training
=====================================
Trains a TF-IDF + Logistic Regression classifier to map operator free-text
to Heli.OS mission types: SAR, SURVEY, PATROL, RECON, MONITOR, ESCORT.

Output:
  intent_classifier.joblib   — LogisticRegression pipeline
  intent_vectorizer.joblib   — TfidfVectorizer (for inference inspection)
  intent_classifier_meta.json

The classifier is consumed by parse_mission_nlp() in mission_orchestrator.py.

Usage:
    python train_intent.py \\
        --data   /tmp/heli-training-data/intent \\
        --output ../../packages/c2_intel/models
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MISSION_TYPES = ["SAR", "SURVEY", "PATROL", "RECON", "MONITOR", "ESCORT"]


def train(data_dir: str, output_dir: str) -> None:
    try:
        import joblib
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        raise RuntimeError("scikit-learn, joblib, pandas required")

    data = Path(data_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = data / "intent_training.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Intent training data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d intent examples", len(df))

    # Validate
    df = df[df["mission_type"].isin(_MISSION_TYPES)].dropna()
    df = df[df["text"].str.strip().str.len() > 3]
    logger.info("After filtering: %d examples", len(df))

    counts = df["mission_type"].value_counts().to_dict()
    logger.info("Class distribution: %s", counts)

    X = df["text"].values
    y = df["mission_type"].values

    # Build TF-IDF + LR pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),       # unigrams, bigrams, trigrams
            max_features=15000,
            sublinear_tf=True,        # log(tf) scaling
            min_df=2,
            analyzer="word",
            token_pattern=r"(?u)\b\w[\w']+\b",
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=500,
            class_weight="balanced",  # handle class imbalance
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )),
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    logger.info("CV F1-macro: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # Full fit
    pipe.fit(X, y)

    # Held-out evaluation (last 20% as pseudo-test)
    split_idx = int(len(df) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    if len(X_test) > 0:
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info("Test classification report:\n%s",
                    classification_report(y_test, y_pred))
    else:
        report = {}

    # Save classifier pipeline (includes vectorizer)
    clf_path = out / "intent_classifier.joblib"
    joblib.dump(pipe, clf_path)
    logger.info("Saved classifier → %s", clf_path.name)

    # Save vectorizer separately for inspection
    vec_path = out / "intent_vectorizer.joblib"
    joblib.dump(pipe.named_steps["tfidf"], vec_path)

    # Top features per class
    classes = pipe.named_steps["clf"].classes_.tolist()
    feature_names = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_
    top_features = {}
    for i, cls in enumerate(classes):
        top_idx = np.argsort(coefs[i])[-10:][::-1]
        top_features[cls] = [feature_names[j] for j in top_idx]

    meta = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "n_samples":         len(df),
        "mission_types":     _MISSION_TYPES,
        "cv_f1_macro_mean":  round(float(cv_scores.mean()), 3),
        "cv_f1_macro_std":   round(float(cv_scores.std()), 3),
        "class_distribution": {k: int(v) for k, v in counts.items()},
        "top_features_per_class": top_features,
        "report":            {k: v for k, v in report.items() if isinstance(v, dict)},
        "source": "FEMA incident data + NIMS/ICS template augmentation",
    }
    meta_path = out / "intent_classifier_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Intent classifier training complete. CV F1: %.3f", cv_scores.mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train mission intent classifier")
    p.add_argument("--data",   required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    train(args.data, args.output)
