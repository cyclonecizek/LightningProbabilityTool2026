"""
prediction.py
-------------
Loads the trained RandomForest models and returns lightning probability (%).

Each model expects features in a FIXED column order (positional — scikit-learn
matches by position when given a NumPy array):

    predict / predict_limited : [thompson_index, wind_average, rh]
    predict_10Z_updated       : [thompson_index, wind_average, rh]
    predict_15Z               : [thompson_index, wind_average, pwat_mm, rh]

predict_proba(...)[:, 1] is the probability of the positive (lightning) class.

NOTE: models are loaded once at import and cached, rather than reloaded on
every call (the original code reloaded from disk on each button press).
If a .sav fails to load (e.g. scikit-learn version mismatch), the error is
raised lazily on first use with a clear message.
"""

import joblib
import pandas as pd

_MODELS = {}

_PATHS = {
    "rfc": "RFC_model.sav",
    "limited": "RFC_model_limited_depth.sav",
    "15Z": "RFC_model_15Z.sav",
    "10Z_updated": "RFC_model_limited_depth_10Z_updated.sav",
}

# Exact feature names AND order each model was trained on. These come from the
# training notebook, which renamed CSV columns via `col.replace(' ', '_')`.
# >>> VERIFY against inspect_models.py output (clf.feature_names_in_) and edit
#     here if they differ. Order matters. <<<
FEATURES_10Z = [
    "Thompson_Index",
    "1000-700mb_Average_U-Wind_Component",
    "700-500mb_Average_RH",
]
FEATURES_15Z = [
    "Thompson_Index",
    "1000-700mb_Average_U-Wind_Component",
    "PWAT",
    "700-500mb_Average_RH",
]


def _get(key):
    """Load and cache a model by key."""
    if key not in _MODELS:
        try:
            _MODELS[key] = joblib.load(_PATHS[key])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{_PATHS[key]}'. This is often a "
                f"scikit-learn version mismatch between training and runtime "
                f"— pin the training version in requirements.txt. ({exc})"
            ) from exc
    return _MODELS[key]


def _proba(clf, data, feature_names=None):
    """
    Return P(lightning) as a percentage Series.

    `data` may be:
      - a dict {feature_name: value}, or
      - a 2D array-like (single row) in the model's trained column order.

    If `feature_names` is given, the input is wrapped in a DataFrame with those
    exact column names so scikit-learn matches features BY NAME (not position),
    which avoids silent misalignment and the "missing feature names" warning.
    """
    if feature_names is not None:
        if isinstance(data, dict):
            row = [[data[name] for name in feature_names]]
        else:
            row = data  # assume already a single-row 2D array in correct order
        X = pd.DataFrame(row, columns=feature_names)
    else:
        X = data
    prediction = clf.predict_proba(X)
    return pd.DataFrame(prediction)[1] * 100


def predict(data):
    return _proba(_get("rfc"), data)


def predict_limited(data):
    return _proba(_get("limited"), data)


def predict_15Z(data):
    return _proba(_get("15Z"), data, feature_names=FEATURES_15Z)


def predict_10Z_updated(data):
    return _proba(_get("10Z_updated"), data, feature_names=FEATURES_10Z)
