"""Pima Indians Diabetes: soft-voting ensemble and shared preprocessing."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).with_name("pima-indians-diabetes.csv")
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
TARGET = "Outcome"
# A value of 0 is physiologically impossible for these, so it means "missing".
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data():
    """Accept the headerless original file or a headered one (target 'Class'/'Outcome')."""
    first = str(pd.read_csv(DATA_PATH, header=None, nrows=1).iloc[0, 0]).strip().lower()
    if first in {"pregnancies", "pregnancy"}:
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Class": TARGET, "class": TARGET})
    else:
        df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    return df.dropna(subset=[TARGET])


def add_features(x):
    """Stateless feature engineering (safe to apply before cross-validation)."""
    x = x.copy()
    present_zero_missing = [column for column in ZERO_IS_MISSING if column in x.columns]
    x[present_zero_missing] = x[present_zero_missing].replace(0, np.nan)
    x["Glucose_BMI"] = x["Glucose"] * x["BMI"] / 100
    x["Glucose_per_Age"] = x["Glucose"] / x["Age"]
    x["LogAge"] = np.log(x["Age"])
    return x


FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]


def prepare_features(values):
    """Build model features from a dict or one-row DataFrame of raw inputs."""
    raw = pd.DataFrame([values], columns=FEATURES)
    return add_features(raw)


def build_model():
    # Imputer lives inside each pipeline, so it is fit on training folds only.
    def with_impute(clf, scale=False):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps + [("clf", clf)])

    linear = with_impute(LogisticRegression(C=0.1, max_iter=3000), scale=True)
    gboost = with_impute(GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.04, max_depth=2, subsample=0.7, random_state=42))
    forest = with_impute(RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1))
    extra = with_impute(ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1))
    return VotingClassifier(
        [("logreg", linear), ("gboost", gboost), ("forest", forest), ("extra", extra)],
        voting="soft", weights=[2, 1, 1, 1])


def main():
    df = load_data()
    y = df[TARGET].astype(int)
    # Keep the fitted feature schema identical to the website's prediction form.
    x = add_features(df[FEATURES])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
               "auc": "roc_auc", "f1": "f1"}
    res = cross_validate(build_model(), x, y, cv=cv, scoring=scoring)
    print(f"Samples: {len(df)}  |  positive rate: {y.mean():.3f}  "
          f"|  majority baseline: {max(y.mean(), 1 - y.mean()):.3f}")
    for k in scoring:
        v = res[f"test_{k}"]
        print(f"{k:>18}: {v.mean():.3f} (+/- {v.std():.3f})")


if __name__ == "__main__":
    main()
