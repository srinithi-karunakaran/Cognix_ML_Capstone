"""Cognix ML Studio: default datasets and best-model prediction for three ML tracks."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, r2_score, mean_absolute_error,
                             mean_squared_error, silhouette_score)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, RandomForestClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans

st.set_page_config(page_title="Cognix ML Studio", page_icon="🧠", layout="wide")
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

st.markdown("""
<style>
.block-container {max-width: 1350px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#334e68); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0; font-size:2.15rem;} .hero p {color:#d9e8f5; margin:.45rem 0 0;}
div[data-testid="stMetric"] {background:rgba(128,128,128,.07); border:1px solid rgba(128,128,128,.18); padding:12px; border-radius:12px;}
</style>
<div class="hero"><h1>🧠 Cognix ML Studio</h1><p>Choose a machine-learning track, adjust input values, and predict with the best-performing model.</p></div>
""", unsafe_allow_html=True)

DATASETS = {
    "Regression": ("regression/Life Expectancy Data.csv", "Life expectancy"),
    "Classification": ("classification/WA_Fn-UseC_-Telco-Customer-Churn.csv", "Churn"),
    "Clustering": ("clustering/tmdb_5000_movies.csv", None),
}

@st.cache_data(show_spinner=False)
def load_default_dataset(track):
    relative, _ = DATASETS[track]
    df = pd.read_csv(DATA / relative)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_preprocessor(X, scale=False):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale: steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")

def candidates(track):
    if track == "Regression":
        return {
            "Random Forest Regressor": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2),
            "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "Ridge Regression": Ridge(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, min_samples_leaf=3),
        }
    return {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, min_samples_leaf=3, class_weight="balanced"),
    }

@st.cache_resource(show_spinner="Training and selecting the best model on the bundled dataset...")
def train_best_model(track):
    df = load_default_dataset(track)
    target = DATASETS[track][1]
    if track == "Clustering":
        # Use meaningful numeric movie attributes; exclude IDs and highly skewed budget/revenue.
        cols = [c for c in ["popularity", "runtime", "vote_average", "vote_count"] if c in df.columns]
        X = df[cols].replace([np.inf, -np.inf], np.nan)
        prep = build_preprocessor(X, scale=True)
        Xp = prep.fit_transform(X)
        best_k, best_score, best_model = 3, -1.0, None
        for k in (2, 3, 4, 5, 6):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(Xp)
            score = silhouette_score(Xp, labels, sample_size=min(1500, len(labels)), random_state=42)
            if score > best_score:
                best_k, best_score, best_model = k, score, model
        return {"df": df, "target": None, "features": cols, "preprocessor": prep,
                "model": best_model, "model_name": f"K-Means ({best_k} clusters)",
                "score_name": "Silhouette score", "score": best_score}

    if track == "Regression":
        df[target] = pd.to_numeric(df[target], errors="coerce")
        # Drop identifier-like country names from regression features, but retain development status.
        excluded = [target, "Country"]
        X = df.drop(columns=[c for c in excluded if c in df.columns])
        y = df[target]
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        score_name = "R² score"
        metric = lambda actual, pred: r2_score(actual, pred)
    else:
        target = "Churn"
        excluded = [target, "customerID"]
        X = df.drop(columns=[c for c in excluded if c in df.columns]).copy()
        if "TotalCharges" in X:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        y = df[target].astype(str)
        score_name = "Weighted F1 score"
        metric = lambda actual, pred: f1_score(actual, pred, average="weighted", zero_division=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if track == "Classification" and y.value_counts().min() >= 2 else None)
    best_name, best_score = None, -np.inf
    for name, estimator in candidates(track).items():
        prep = build_preprocessor(X_train, scale=name in ["Ridge Regression", "Logistic Regression"])
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
        score = metric(y_test, pred)
        if np.isfinite(score) and score > best_score:
            best_name, best_score = name, float(score)
    # Refit the single best model on all labelled rows for interactive predictions.
    estimator = candidates(track)[best_name]
    prep = build_preprocessor(X, scale=best_name in ["Ridge Regression", "Logistic Regression"])
    final_pipe = Pipeline([("preprocess", prep), ("model", estimator)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe.fit(X, y)
    return {"df": df, "target": target, "features": X.columns.tolist(), "pipeline": final_pipe,
            "model_name": best_name, "score_name": score_name, "score": best_score}

def default_value(series):
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return float(vals.median()) if not vals.empty else 0.0
    vals = series.dropna().astype(str)
    return str(vals.mode().iloc[0]) if not vals.empty else "Unknown"

def input_form(features, data, key_prefix):
    values = {}
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in features if c not in numeric]
    left, right = st.columns(2)
    for i, col in enumerate(numeric):
        series = pd.to_numeric(data[col], errors="coerce").dropna()
        default = default_value(data[col])
        low = float(series.min()) if not series.empty else default - 1
        high = float(series.max()) if not series.empty else default + 1
        if low == high:
            low, high = low - 1, high + 1
        step = 1.0 if pd.api.types.is_integer_dtype(data[col]) else max((high-low)/100, 0.01)
        with (left if i % 2 == 0 else right):
            values[col] = st.number_input(col, min_value=float(low), max_value=float(high), value=float(np.clip(default, low, high)), step=float(step), key=f"{key_prefix}_{col}")
    for i, col in enumerate(categorical):
        vals = data[col].dropna().astype(str).unique().tolist()
        vals = sorted(vals)[:300]
        default = default_value(data[col])
        if default not in vals and vals: vals.insert(0, default)
        with (left if i % 2 == 0 else right):
            values[col] = st.selectbox(col, vals or ["Unknown"], index=(vals.index(default) if default in vals else 0), key=f"{key_prefix}_{col}")
    return pd.DataFrame([values], columns=features)

with st.sidebar:
    st.markdown("## Workspace")
    track = st.radio("Choose a track", ["Regression", "Classification", "Clustering"], horizontal=False)
    st.divider()
    st.caption("Bundled datasets load automatically. No CSV upload is required.")

try:
    result = train_best_model(track)
except Exception as exc:
    st.error(f"Could not prepare the {track.lower()} track: {exc}")
    st.stop()

df = result["df"]
st.markdown(f"## {track}")
if track == "Regression":
    st.write("Predict a numeric value using the best-performing regression model selected from a held-out test split.")
elif track == "Classification":
    st.write("Estimate whether a customer will churn using the best-performing classifier selected from a held-out test split.")
else:
    st.write("Assign a movie to a discovered group using the best K-Means configuration selected by silhouette score.")

m1, m2, m3 = st.columns(3)
m1.metric("Default dataset", {"Regression":"Life Expectancy", "Classification":"Telco Customer Churn", "Clustering":"TMDB Movies"}[track])
m2.metric("Rows loaded", f"{len(df):,}")
m3.metric("Selected model", result["model_name"])
st.caption(f"Model selection score: {result['score_name']} = {result['score']:.3f}. The score is measured during model selection, not a guarantee for every prediction.")
with st.expander("Preview the default CSV", expanded=False):
    st.dataframe(df.head(12), use_container_width=True)
    st.caption("The dataset is bundled with the project and loaded automatically when you open this track.")

st.divider()
if track in ("Regression", "Classification"):
    st.markdown("### Enter values to predict")
    st.caption("Input fields start with typical values from the dataset. Adjust them to describe the case you want to predict.")
    X = input_form(result["features"], df, track.lower())
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        try:
            pipe = result["pipeline"]
            prediction = pipe.predict(X)[0]
            if track == "Regression":
                st.success(f"Predicted life expectancy: **{float(prediction):.2f} years**")
            else:
                st.success(f"Predicted customer churn: **{prediction}**")
                model = pipe.named_steps["model"]
                if hasattr(model, "predict_proba"):
                    probs = pipe.predict_proba(X)[0]
                    classes = list(model.classes_)
                    if "Yes" in classes:
                        prob = float(probs[classes.index("Yes")])
                        st.metric("Churn probability", f"{prob:.1%}")
                        st.progress(prob)
            with st.expander("Review the input values"):
                st.dataframe(X, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.markdown("### Enter movie characteristics")
    st.caption("Adjust numeric movie details. The selected model assigns the movie to the nearest learned cluster.")
    X = input_form(result["features"], df, "cluster")
    if st.button("🧩 Predict movie cluster", type="primary", use_container_width=True):
        try:
            transformed = result["preprocessor"].transform(X)
            cluster = int(result["model"].predict(transformed)[0])
            st.success(f"This movie belongs to **Cluster {cluster}**")
            centers = result["model"].cluster_centers_
            distances = np.linalg.norm(transformed - centers[cluster], axis=1)
            st.metric("Distance from cluster centre", f"{float(distances[0]):.3f}", help="Smaller distance means the input is closer to this cluster centre in scaled feature space.")
            summary = pd.DataFrame({"Feature": result["features"], "Your value": [X.iloc[0][c] for c in result["features"]]})
            with st.expander("Review the input values"):
                st.dataframe(summary, use_container_width=True)
        except Exception as exc:
            st.error(f"Cluster prediction failed: {exc}")

st.divider()
st.caption("Cognix ML Studio · Default datasets load automatically · One selected best model per track")
