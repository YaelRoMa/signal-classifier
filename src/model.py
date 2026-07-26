"""
model.py

Trains and evaluates simple classifiers (logistic regression and
decision tree) on the features extracted from the synthetic signals.

Typical usage:
    python model.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from generate_signals import generate_dataset
from extract_features import extract_features, FEATURE_NAMES


def prepare_data(n_per_class=200, seed=42, test_size=0.25):
    X_time, y, meta = generate_dataset(n_per_class=n_per_class, seed=seed)
    X_feat = extract_features(X_time)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{'=' * 50}")
    print(f"Model: {model_name}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, y_pred


def main():
    X_train, X_test, y_train, y_test, scaler = prepare_data(n_per_class=200)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    }

    results = {}
    for name, model in models.items():
        trained_model, y_pred = train_and_evaluate(
            model, name, X_train, X_test, y_train, y_test
        )
        results[name] = trained_model

    # Feature relevance for the decision tree
    tree = results["Decision Tree"]
    print(f"\n{'=' * 50}")
    print("Feature importance (Decision Tree):")
    print(f"{'=' * 50}")
    for feat_name, importance in zip(FEATURE_NAMES, tree.feature_importances_):
        print(f"  {feat_name:20s}: {importance:.3f}")

    return results


if __name__ == "__main__":
    main()