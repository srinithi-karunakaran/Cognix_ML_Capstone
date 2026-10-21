"""Cognix ML Studio: CSV-driven regression, classification and clustering."""
from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

st.set_page_config(page_title="Cognix ML Studio", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#334e68); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0; font-size:2.2rem;}
.hero p {color:#d9e8f5; margin:.45rem 0 0 0;}
div[data-testid="stMetric"] {background:rgba(128,128,128,.07); border:1px solid rgba(128,128,128,.18); padding:14px; border-radius:12px;}
</style>
<div class="hero"><h1>🧠 Cognix ML Studio</h1><p>One workspace for Regression, Classification, and Clustering — upload a CSV, choose an algorithm, and explore the results.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.caption("CSV-based machine learning")
    task = st.radio("Choose a task", ["Regression", "Classification", "Clustering"], index=0)
    st.divider()
    st.markdown("**Workflow**")
    st.markdown("1. Upload CSV\n2. Choose columns\n3. Select algorithm\n4. Run analysis\n5. Download results")
    st.caption("Your uploaded data is processed for this app session.")

@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise ValueError("Choose at least one usable feature column.")
    return ColumnTransformer(transformers, remainder="drop"), numeric, categorical

def estimator_options(task_name: str):
    if task_name == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(max_iter=5000),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Regressor": SVR(),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Classifier": SVC(),
    }

sample_col1, sample_col2 = st.columns([3, 2])
with sample_col1:
    uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"], help="Use a tabular CSV file with column headers.")
with sample_col2:
    st.markdown("**No dataset yet?**")
    st.caption("Load a small sample dataset to test the interface.")
    sample_type = st.selectbox("Sample dataset", ["Regression sample", "Classification sample", "Clustering sample"], label_visibility="collapsed")
    load_sample = st.button("Load sample dataset", use_container_width=True)

if "sample_frame" not in st.session_state:
    st.session_state.sample_frame = None
if load_sample:
    rng = np.random.default_rng(42)
    if sample_type == "Regression sample":
        x = np.arange(1, 101)
        st.session_state.sample_frame = pd.DataFrame({"study_hours": x / 10, "attendance": rng.integers(60, 101, 100), "score": 25 + 6 * x / 10 + rng.normal(0, 7, 100)})
    elif sample_type == "Classification sample":
        n = 120
        age = rng.integers(18, 75, n)
        spend = rng.normal(65, 20, n).clip(10, 120)
        risk = ((age > 55).astype(int) + (spend < 45).astype(int) + rng.integers(0, 2, n)) >= 2
        st.session_state.sample_frame = pd.DataFrame({"age": age, "monthly_spend": spend.round(2), "customer_group": np.where(risk, "At risk", "Regular")})
    else:
        a = rng.normal(loc=[2, 2], scale=.55, size=(80, 2))
        b = rng.normal(loc=[6, 6], scale=.65, size=(80, 2))
        c = rng.normal(loc=[2, 7], scale=.5, size=(80, 2))
        arr = np.vstack([a, b, c])
        st.session_state.sample_frame = pd.DataFrame(arr, columns=["feature_x", "feature_y"])

if uploaded is not None:
    try:
        df = read_csv_bytes(uploaded.getvalue())
        st.session_state.sample_frame = df
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
else:
    df = st.session_state.sample_frame

if df is None:
    st.info("Upload a CSV or load a sample dataset to begin.")
    st.markdown("### What you can do")
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 📈 Regression\nPredict a numeric target and compare model errors.")
    c2.markdown("#### 🏷️ Classification\nPredict categories and review precision, recall, F1, and confusion matrix.")
    c3.markdown("#### 🧩 Clustering\nFind groups in your data without a target column.")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The CSV must contain data and at least two columns.")
    st.stop()

# Basic data overview
st.markdown("### Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
with st.expander("Preview data and column types", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)
    st.dataframe(pd.DataFrame({"Column": df.columns, "Data type": [str(t) for t in df.dtypes], "Missing": [int(df[c].isna().sum()) for c in df.columns]}), hide_index=True, use_container_width=True)

st.divider()
st.markdown(f"### {task} setup")

if task in ("Regression", "Classification"):
    target_candidates = df.select_dtypes(include=np.number).columns.tolist() if task == "Regression" else df.columns.tolist()
    if not target_candidates:
        st.error("Regression requires at least one numeric target column.")
        st.stop()
    default_target = target_candidates[-1]
    target = st.selectbox("Target column to predict", target_candidates, index=target_candidates.index(default_target))
    feature_candidates = [c for c in df.columns if c != target]
    default_features = feature_candidates
    features = st.multiselect("Input feature columns", feature_candidates, default=default_features)
    a, b = st.columns(2)
    with a:
        algorithm = st.selectbox("Algorithm", list(estimator_options(task).keys()))
    with b:
        test_size = st.slider("Test data (%)", 10, 40, 20, 5) / 100
    if st.button(f"🚀 Run {task.lower()}", type="primary", use_container_width=True):
        if not features:
            st.error("Select at least one input feature.")
            st.stop()
        working = df[features + [target]].copy().dropna(subset=[target])
        X = working[features]
        y = working[target]
        if task == "Regression":
            y = pd.to_numeric(y, errors="coerce")
            valid = y.notna()
            X, y = X.loc[valid], y.loc[valid]
            if len(y) < 5:
                st.error("Regression needs at least 5 rows with numeric target values.")
                st.stop()
        else:
            y = y.astype(str)
            if y.nunique() < 2:
                st.error("Classification target must contain at least two classes.")
                st.stop()
            counts = y.value_counts()
            stratify = y if counts.min() >= 2 else None
            if stratify is None:
                st.warning("Some classes have only one row, so the split will not be stratified.")
        try:
            stratify = None
            if task == "Classification" and y.value_counts().min() >= 2:
                stratify = y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
            preprocessor, _, _ = make_preprocessor(X_train, scale_numeric=algorithm in ["Logistic Regression", "K-Nearest Neighbors Classifier", "Support Vector Classifier", "Ridge Regression", "Lasso Regression", "K-Nearest Neighbors Regressor", "Support Vector Regressor"])
            estimator = estimator_options(task)[algorithm]
            pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            with st.spinner("Training model and evaluating held-out data..."):
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
            st.success(f"{algorithm} trained successfully.")
            if task == "Regression":
                mae = mean_absolute_error(y_test, predictions)
                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
                r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")
                k1, k2, k3 = st.columns(3)
                k1.metric("MAE ↓", f"{mae:.4f}")
                k2.metric("RMSE ↓", f"{rmse:.4f}")
                k3.metric("R² score ↑", f"{r2:.4f}")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(y_test, predictions, alpha=.7)
                lo = float(min(np.min(y_test), np.min(predictions)))
                hi = float(max(np.max(y_test), np.max(predictions)))
                ax.plot([lo, hi], [lo, hi], linestyle="--")
                ax.set_xlabel("Actual values"); ax.set_ylabel("Predicted values"); ax.set_title("Actual vs predicted")
                st.pyplot(fig); plt.close(fig)
            else:
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
                recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
                f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Accuracy", f"{accuracy:.3f}"); k2.metric("Precision (weighted)", f"{precision:.3f}")
                k3.metric("Recall (weighted)", f"{recall:.3f}"); k4.metric("F1 (weighted)", f"{f1:.3f}")
                labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(predictions)] )).tolist())
                cm = confusion_matrix(y_test, predictions, labels=labels)
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_title("Confusion matrix"); ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
                ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels=labels)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04); fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                st.text("Classification report\n" + classification_report(y_test, predictions, zero_division=0))
            output = X_test.copy()
            output[f"Actual {target}"] = y_test.values
            output[f"Predicted {target}"] = predictions
            st.markdown("#### Predictions on test data")
            st.dataframe(output, use_container_width=True)
            st.download_button("⬇️ Download test predictions (CSV)", output.to_csv(index=False).encode("utf-8"), file_name=f"{task.lower()}_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not run this model: {exc}")
            st.info("Try removing identifier columns, choosing different features, or selecting another algorithm.")
else:
    feature_candidates = df.columns.tolist()
    numeric_candidates = df.select_dtypes(include=np.number).columns.tolist()
    default_features = numeric_candidates if len(numeric_candidates) >= 2 else feature_candidates
    features = st.multiselect("Features to group", feature_candidates, default=default_features)
    a, b = st.columns(2)
    with a:
        cluster_algorithm = st.selectbox("Clustering algorithm", ["K-Means", "Agglomerative Clustering", "DBSCAN"])
    with b:
        if cluster_algorithm in ("K-Means", "Agglomerative Clustering"):
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
        else:
            eps = st.slider("DBSCAN epsilon", 0.1, 3.0, 0.5, 0.1)
            min_samples = st.slider("DBSCAN min samples", 2, 20, 5)
    if st.button("🧩 Find clusters", type="primary", use_container_width=True):
        if not features:
            st.error("Select at least one feature.")
            st.stop()
        try:
            X = df[features].copy()
            # Drop columns that contain no values and rows that have no feature information.
            X = X.dropna(axis=1, how="all")
            if X.shape[1] == 0:
                st.error("The selected features contain no usable values.")
                st.stop()
            preprocessor, _, _ = make_preprocessor(X, scale_numeric=True)
            X_processed = preprocessor.fit_transform(X)
            if X_processed.shape[0] < 3:
                st.error("Clustering needs at least three rows.")
                st.stop()
            if cluster_algorithm == "K-Means":
                labels = KMeans(n_clusters=min(n_clusters, len(X) - 1), random_state=42, n_init=10).fit_predict(X_processed)
            elif cluster_algorithm == "Agglomerative Clustering":
                labels = AgglomerativeClustering(n_clusters=min(n_clusters, len(X) - 1)).fit_predict(X_processed)
            else:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_processed)
            output = df.copy()
            output["Cluster"] = labels
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = int(np.sum(labels == -1))
            k1, k2, k3 = st.columns(3)
            k1.metric("Clusters found", n_found); k2.metric("Rows grouped", int(np.sum(labels != -1))); k3.metric("Noise points", noise_count)
            if n_found >= 2 and n_found < len(labels):
                try:
                    score = silhouette_score(X_processed, labels)
                    st.metric("Silhouette score", f"{score:.3f}", help="Higher is generally better; noise points are excluded for DBSCAN.")
                except Exception:
                    pass
            st.markdown("#### Cluster sizes")
            sizes = output["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Rows")
            st.bar_chart(sizes.set_index("Cluster"))
            if len(numeric_candidates) >= 2:
                x_axis = st.selectbox("Plot X axis", numeric_candidates, index=0)
                y_axis = st.selectbox("Plot Y axis", [c for c in numeric_candidates if c != x_axis], index=0)
                plot_df = output[[x_axis, y_axis, "Cluster"]].copy()
                plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors="coerce")
                plot_df[y_axis] = pd.to_numeric(plot_df[y_axis], errors="coerce")
                plot_df = plot_df.dropna()
                fig, ax = plt.subplots(figsize=(8, 5))
                for label in sorted(plot_df["Cluster"].unique()):
                    group = plot_df[plot_df["Cluster"] == label]
                    ax.scatter(group[x_axis], group[y_axis], label=f"Cluster {label}", alpha=.75)
                ax.set_xlabel(x_axis); ax.set_ylabel(y_axis); ax.set_title("Cluster visualization"); ax.legend()
                st.pyplot(fig); plt.close(fig)
            st.markdown("#### Data with cluster labels")
            st.dataframe(output, use_container_width=True)
            st.download_button("⬇️ Download clustered data (CSV)", output.to_csv(index=False).encode("utf-8"), file_name="clustered_data.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not run clustering: {exc}")
            st.info("Try selecting fewer columns or numeric columns with sufficient non-missing values.")

st.divider()
st.caption("Cognix ML Studio · Educational use · Results depend on the selected dataset, features, and algorithm.")
