"""
Machine Learning models definition, hyperparameter evaluation, and serialization.
Includes SVM (RBF & Linear), Random Forest, Logistic Regression, and KNN.
"""

import os
import pickle
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_candidate_models() -> Dict[str, Any]:
    """Returns a dictionary of candidate classifiers for Indian Cricketer Face Identification."""
    return {
        "SVM (RBF Kernel)": SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=42
        ),
        "SVM (Linear Kernel)": SVC(
            C=1.0,
            kernel="linear",
            probability=True,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_split=2,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="cosine"
        )
    }


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5
) -> Tuple[Dict[str, Dict[str, float]], str, Any]:
    """
    Trains and benchmarks candidate models using cross-validation and test set evaluation.
    Returns:
        - results_dict: performance metrics for each model
        - best_model_name: key of best model
        - best_estimator: fitted best model instance
    """
    models = get_candidate_models()
    results = {}
    best_score = -1.0
    best_model_name = ""
    best_estimator = None

    skf = StratifiedKFold(n_splits=min(cv_folds, len(np.unique(y_train))), shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"[*] Training {name}...")

        # 5-fold cross validation on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

        # Fit model on full training set
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "cv_mean_acc": float(np.mean(cv_scores)),
            "cv_std_acc": float(np.std(cv_scores)),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "model_obj": model
        }

        print(f"    -> CV Acc: {np.mean(cv_scores):.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")

        # Choose best based on combination of CV and test accuracy
        combined_score = 0.5 * np.mean(cv_scores) + 0.5 * test_acc
        if combined_score > best_score:
            best_score = combined_score
            best_model_name = name
            best_estimator = model

    print(f"\n[+] Champion Model: {best_model_name} (Combined Score: {best_score:.4f})")
    return results, best_model_name, best_estimator


def save_artifacts(
    model: Any,
    scaler: Any,
    class_mapping: Dict[int, str],
    metrics: Dict[str, Any],
    artifacts_dir: str = "artifacts"
):
    """Saves trained model, scaler, class mappings, and metrics to disk."""
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(os.path.join(artifacts_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(artifacts_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    import json
    with open(os.path.join(artifacts_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=4)

    # Clean non-serializable objects from metrics before json dumping
    clean_metrics = {}
    for k, v in metrics.items():
        clean_metrics[k] = {ik: iv for ik, iv in v.items() if ik != "model_obj"}

    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(clean_metrics, f, indent=4)

    print(f"[+] Saved model artifacts to '{artifacts_dir}/'")


def load_artifacts(artifacts_dir: str = "artifacts") -> Tuple[Any, Any, Dict[int, str]]:
    """Loads trained model, scaler, and class mapping from disk."""
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    mapping_path = os.path.join(artifacts_dir, "class_mapping.json")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing artifacts in '{artifacts_dir}'. Please run train.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    import json
    with open(mapping_path, "r") as f:
        # Convert string keys back to int
        raw_mapping = json.load(f)
        class_mapping = {int(k): v for k, v in raw_mapping.items()}

    return model, scaler, class_mapping

