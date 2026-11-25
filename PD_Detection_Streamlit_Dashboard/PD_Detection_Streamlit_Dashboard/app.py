from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
DATA_FILE = DATA_DIR / "parkinsons.data"

st.set_page_config(page_title="Parkinson's Detection Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b1020; color: #e8edf7; }
[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #293449; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #f4f7ff; }
p, label, .stMarkdown { color: #d5dced; }
div[data-testid="stMetric"] { background: #151e31; border: 1px solid #2c3951; padding: 14px 16px; border-radius: 12px; }
div[data-testid="stMetricLabel"] { color: #aebbd2; }
div[data-testid="stMetricValue"] { color: #f8fafc; }
.card { background: #151e31; border: 1px solid #2c3951; padding: 18px; border-radius: 14px; margin-bottom: 12px; }
.small-muted { color: #aebbd2; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

def load_dataset():
    if not DATA_FILE.exists():
        with st.spinner("Downloading the UCI Parkinson's voice dataset…"):
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    if not {"name", "status"}.issubset(df.columns):
        raise ValueError("Dataset does not contain the expected name/status columns.")
    groups = df["name"].astype(str).str.extract(r"_(S\d+)_", expand=False)
    if groups.isna().any():
        raise ValueError("Could not identify participants from the dataset name column.")
    y = pd.to_numeric(df["status"], errors="raise").astype(int)
    X = df.drop(columns=["name", "status"]).apply(pd.to_numeric, errors="coerce")
    return df, X, y, groups

def make_models():
    def pipe(estimator, scale=True):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)
    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        "SVM": pipe(SVC(probability=True, class_weight="balanced", random_state=42)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42), scale=False),
        "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=42), scale=False),
    }

@st.cache_data(show_spinner=False)
def get_dataset_cached():
    return load_dataset()

@st.cache_resource(show_spinner=False)
def train_research_models():
    df, X, y, groups = load_dataset()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(X, y, groups))
    train_idx, test_idx = splits[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
    if set(g_train) & set(g_test):
        raise RuntimeError("Participant overlap detected in the train/test split.")
    models = make_models()
    fitted, rows, train_rows, timing_rows = {}, [], [], []
    for name, est in models.items():
        start = time.perf_counter()
        est.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        train_pred = est.predict(X_train)
        start = time.perf_counter()
        pred = est.predict(X_test)
        predict_time = time.perf_counter() - start
        try:
            prob = est.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, prob)
        except Exception:
            auc = np.nan
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc
        })
        train_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_train, train_pred),
            "Precision": precision_score(y_train, train_pred, zero_division=0),
            "Recall": recall_score(y_train, train_pred, zero_division=0),
            "F1-score": f1_score(y_train, train_pred, zero_division=0),
        })
        timing_rows.append({"Model": name, "Training time (seconds)": fit_time,
                            "Test prediction time (seconds)": predict_time})
        fitted[name] = est
    test_df = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
    return df, X, y, groups, X_train, X_test, y_train, y_test, fitted, test_df, pd.DataFrame(train_rows), pd.DataFrame(timing_rows)

def load_csv(name):
    path = RESULTS_DIR / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def title_block(title, subtitle):
    st.title(title)
    st.caption(subtitle)

with st.sidebar:
    st.markdown("## 🧠 PD Detection Lab")
    st.caption("Research dashboard · Voice baseline")
    page = st.radio("NAVIGATION", [
        "Home", "Dataset Analysis", "Prediction", "Model Comparison",
        "Performance Dashboard", "Confusion Matrix", "Research Results"
    ])
    st.divider()
    st.markdown("**Project status**")
    st.success("Voice baseline configured")
    st.warning("Multimodal fusion not implemented yet")
    st.caption("For research demonstration only — not a medical diagnosis.")

try:
    with st.spinner("Loading dataset and evaluating research models…"):
        df, X, y, groups, X_train, X_test, y_train, y_test, fitted, test_df, train_df, timings = train_research_models()
except Exception as exc:
    st.error(f"Could not initialize the dataset/model pipeline: {exc}")
    st.info("Check your internet connection. The first launch downloads the UCI voice dataset into the data folder.")
    st.stop()

best = test_df.iloc[0]
n_pd = int((y == 1).sum())
n_control = int((y == 0).sum())

if page == "Home":
    title_block("Parkinson's Disease Detection", "An interactive machine-learning research dashboard inspired by your reference screenshots.")
    st.markdown('<div class="card"><h3>Research workspace</h3><p>Explore the UCI voice dataset, compare five machine-learning baselines, review measured performance, and run an example prediction from voice-feature values.</p><p class="small-muted">Current implementation: voice-only baseline. Multimodal fusion needs compatible, participant-linked data.</p></div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Voice recordings", f"{len(df):,}")
    b.metric("Participants", f"{groups.nunique():,}")
    c.metric("Input features", f"{X.shape[1]}")
    d.metric("Best F1 in current run", f"{best['F1-score']:.3f}")
    st.subheader("Model snapshot")
    st.dataframe(test_df.round(4), use_container_width=True, hide_index=True)
    st.subheader("Getting started")
    st.markdown("Use the left sidebar to explore dataset statistics, model comparison, confusion matrix, and the research tables.")
    st.info("Performance is based on a participant-grouped holdout split. Results can vary with split choices and small datasets.")

elif page == "Dataset Analysis":
    title_block("Dataset Analysis", "Understand the class balance and numeric voice features.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total records", len(df))
    c2.metric("Control records", n_control)
    c3.metric("Parkinson's records", n_pd)
    left, right = st.columns(2)
    with left:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        counts = y.value_counts().reindex([0,1], fill_value=0)
        ax.bar(["Control (0)", "Parkinson's (1)"], counts.values, color=["#aebbd2", "#4f83cc"], edgecolor="black")
        ax.set_ylabel("Record count")
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with right:
        st.subheader("Feature summary")
        st.dataframe(X.describe().T.round(3), use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(df.drop(columns=["name"]).head(12), use_container_width=True, hide_index=True)
    st.caption("The dataset contains repeated recordings per participant; this dashboard groups participant IDs for the held-out evaluation.")

elif page == "Prediction":
    title_block("Voice Feature Prediction", "Enter numerical voice measurements to run a research-model demonstration.")
    st.warning("This is not a clinical diagnostic test. The result is only a model output for research/education.")
    model_name = st.selectbox("Model", list(fitted.keys()), index=list(fitted.keys()).index("KNN") if "KNN" in fitted else 0)
    st.markdown("Enter values from a compatible voice-feature record. Defaults are medians from the dataset, not patient-specific measurements.")
    defaults = X.median(numeric_only=True)
    feature_names = list(X.columns)
    with st.form("prediction_form"):
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                val = float(defaults[feature]) if pd.notna(defaults[feature]) else 0.0
                values[feature] = st.number_input(feature, value=val, format="%.6f", key=f"feature_{feature}")
        submitted = st.form_submit_button("Run research prediction", type="primary")
    if submitted:
        row = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
        model = fitted[model_name]
        pred = int(model.predict(row)[0])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(row)[0,1])
        else:
            prob = np.nan
        if pred == 1:
            st.error("Model output: Parkinson's class (1)")
        else:
            st.success("Model output: Control class (0)")
        if np.isfinite(prob):
            st.metric("Model-estimated class-1 probability", f"{prob:.1%}")
        st.caption("This output is not a diagnosis, a risk estimate, or a substitute for assessment by a qualified clinician.")

elif page == "Model Comparison":
    title_block("Model Comparison", "Compare held-out performance from the current participant-grouped experiment.")
    metric = st.selectbox("Metric to compare", ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    chart_df = test_df.set_index("Model")[[metric]].sort_values(metric, ascending=False)
    st.bar_chart(chart_df, y=metric)
    st.dataframe(test_df.round(4), use_container_width=True, hide_index=True)
    st.caption("The displayed metrics are computed by this run; they are not manually entered.")

elif page == "Performance Dashboard":
    title_block("Performance Dashboard", "Measured test metrics and basic compute timing.")
    selected_metric = st.selectbox("Primary metric", ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], index=3)
    selected = test_df.sort_values(selected_metric, ascending=False).iloc[0]
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Top model", selected["Model"])
    m2.metric(f"Best {selected_metric}", f"{selected[selected_metric]:.3f}")
    m3.metric("Train records", len(X_train))
    m4.metric("Test records", len(X_test))
    st.subheader("Metrics by model")
    st.dataframe(test_df.round(4), use_container_width=True, hide_index=True)
    st.subheader("Training and prediction time")
    st.dataframe(timings.round(4), use_container_width=True, hide_index=True)
    st.caption(f"Python environment uses scikit-learn {sklearn.__version__}. Runtime measurements depend on hardware and current system load.")

elif page == "Confusion Matrix":
    title_block("Confusion Matrix", "Inspect the selected model's actual held-out predictions.")
    model_name = st.selectbox("Model", list(fitted.keys()), index=list(fitted.keys()).index("KNN") if "KNN" in fitted else 0)
    pred = fitted[model_name].predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(cm, display_labels=["Control", "Parkinson's"]).plot(ax=ax, cmap="Greys", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("Counts are from the held-out test set in this run.")

elif page == "Research Results":
    title_block("Research Results", "Review tables produced by the implementation.")
    tables = [
        ("Table 6.1 — Dataset characteristics", "table_6_1_dataset.csv"),
        ("Table 6.2 — Class distribution", "table_6_2_class_distribution.csv"),
        ("Table 6.3 — Experimental environment", "table_6_3_environment.csv"),
        ("Table 6.4 — Preprocessing", "table_6_4_preprocessing.csv"),
        ("Table 6.5 — Training performance", "table_6_5_training_performance.csv"),
        ("Table 6.6 — Test performance", "table_6_6_test_performance.csv"),
        ("Table 6.7 — Cross-validation", "table_6_7_cross_validation.csv"),
        ("Table 6.8 — Computational comparison", "table_6_8_computational_comparison.csv"),
    ]
    st.caption("Existing CSV files are shown when present. If a file is missing, the app shows live metrics where possible.")
    for label, fname in tables:
        with st.expander(label):
            saved = load_csv(fname)
            if saved is not None:
                st.dataframe(saved, use_container_width=True, hide_index=True)
                st.download_button(f"Download {fname}", saved.to_csv(index=False).encode("utf-8"), file_name=fname, mime="text/csv", key=f"download_{fname}")
            elif fname == "table_6_6_test_performance.csv":
                st.dataframe(test_df.round(4), use_container_width=True, hide_index=True)
            elif fname == "table_6_5_training_performance.csv":
                st.dataframe(train_df.round(4), use_container_width=True, hide_index=True)
            elif fname == "table_6_8_computational_comparison.csv":
                st.dataframe(timings.round(4), use_container_width=True, hide_index=True)
            elif fname == "table_6_1_dataset.csv":
                st.dataframe(pd.DataFrame([{"Dataset":"UCI Oxford Parkinson's voice dataset","Records":len(df),"Unique participants":groups.nunique(),"Input features":X.shape[1],"Modality":"Voice only"}]), use_container_width=True, hide_index=True)
            elif fname == "table_6_2_class_distribution.csv":
                st.dataframe(pd.DataFrame({"Class":["Control (0)","Parkinson's disease (1)"],"Count":[n_control,n_pd],"Percentage":[round(n_control/len(y)*100,2),round(n_pd/len(y)*100,2)]}), use_container_width=True, hide_index=True)
            else:
                st.info("This saved table is not present in the results folder yet. Run the research pipeline to generate it.")
    st.download_button("Download current test metrics CSV", test_df.to_csv(index=False).encode("utf-8"), file_name="current_test_metrics.csv", mime="text/csv")

st.divider()
st.caption("Parkinson's Disease Detection Lab · Educational research prototype · Voice-only baseline · Not for clinical use")
