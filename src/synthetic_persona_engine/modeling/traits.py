"""Trait inference models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TraitModel:
    feature_columns: list[str]
    label_column: str
    classifier: LogisticRegression
    scaler: StandardScaler

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(features[self.feature_columns])
        return self.classifier.predict_proba(X)[:, 1]


@dataclass
class TraitEffect:
    treatment_effect: float
    confidence_interval: tuple[float, float]


def train_trait_classifier(
    features: pd.DataFrame,
    *,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TraitModel:
    """Train a logistic regression classifier to predict persona traits."""

    feature_columns = [col for col in features.columns if col != label_column]
    X = features[feature_columns]
    y = features[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_scaled, y_train)

    accuracy = clf.score(X_test_scaled, y_test)
    print(f"Trait classifier accuracy: {accuracy:.3f}")

    return TraitModel(
        feature_columns=feature_columns,
        label_column=label_column,
        classifier=clf,
        scaler=scaler,
    )


def estimate_trait_effect(
    data: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    feature_columns: Optional[list[str]] = None,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> TraitEffect:
    """Estimate the causal effect of a trait on an outcome using stratified bootstrap."""

    rng = np.random.default_rng(random_state)
    feature_columns = feature_columns or [
        col for col in data.columns if col not in {treatment_column, outcome_column}
    ]

    # Propensity-score weighting via logistic regression.
    propensity_model = LogisticRegression(max_iter=200)
    propensity_model.fit(data[feature_columns], data[treatment_column])
    propensity_scores = propensity_model.predict_proba(data[feature_columns])[:, 1]

    treated = data[treatment_column] == 1
    control = ~treated

    weights_treated = 1 / propensity_scores[treated]
    weights_control = 1 / (1 - propensity_scores[control])

    te = (
        np.average(data.loc[treated, outcome_column], weights=weights_treated)
        - np.average(data.loc[control, outcome_column], weights=weights_control)
    )

    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(data), size=len(data), replace=True)
        sample = data.iloc[idx]
        treated_mask = sample[treatment_column] == 1
        control_mask = ~treated_mask
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            continue
        est = sample.loc[treated_mask, outcome_column].mean() - sample.loc[control_mask, outcome_column].mean()
        estimates.append(est)

    if not estimates:
        estimates = [te]

    ci_low, ci_high = np.percentile(estimates, [2.5, 97.5])
    return TraitEffect(treatment_effect=float(te), confidence_interval=(float(ci_low), float(ci_high)))
