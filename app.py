"""ProphetBoost Studio — electricity load forecasting dashboard.

A simple forecasting workspace: pick a dataset, pick models, run them,
and watch the forecast unfold. Built on the real ProphetBoost research
project (Prophet + stability-selected XGBoost) plus a roster of common
ML models so anyone can compare forecasting approaches on real data.

Run with: streamlit run app.py
"""

import json
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymupdf
import streamlit as st
import xgboost as xgb
from prophet import Prophet
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
LOAD_CSV = BASE_DIR / "totalload_new.csv"
UPLOAD_DIR = BASE_DIR / "uploaded_datasets"
MANIFEST_PATH = UPLOAD_DIR / "manifest.json"
FEATURE_IMPORTANCE_IMG = BASE_DIR / "feature_importance.png"
WEATHER_PLOTS_IMG = BASE_DIR / "overlaid_weather_plots.png"
FIG_PIPELINE = BASE_DIR / "fig-11.pdf"
FIG_FEATURE_SELECTION = BASE_DIR / "fig-3.pdf"
FIG_SELECTION_OUTCOME = BASE_DIR / "fig-1.png"
FIG_MODEL_DIAGNOSTICS = BASE_DIR / "fig-2.png"
FIG_SHAP_BEESWARM = BASE_DIR / "fig-7 (1).png"
UPLOAD_DIR.mkdir(exist_ok=True)

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
SERIES_YELLOW = "#eda100"
SERIES_MAGENTA = "#e87ba4"
SERIES_VIOLET = "#4a3aa7"
SERIES_GREEN = "#008300"
SERIES_RED = "#e34948"
SERIES_TEAL = "#0f9b8e"
SERIES_BROWN = "#a0522d"
CATEGORICAL = [SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA, SERIES_YELLOW, SERIES_MAGENTA,
               SERIES_VIOLET, SERIES_GREEN, SERIES_RED, SERIES_TEAL, SERIES_BROWN]

PAPER = {
    "title": "ProphetBoost: a hybrid pipeline for accurate and transparent electricity load forecasting",
    "authors": "Nazrul Amin, Yong-Woon Kim, Yung-Cheol Byun",
    "venue": "Computers & Electrical Engineering",
    "citation": "Computers and Electrical Engineering 138 (2026) 111280",
    "doi": "10.1016/j.compeleceng.2026.111280",
    "affiliation": "Artificial Intelligence Lab, Jeju National University",
    "advisor": "Prof. Yung-Cheol Byun",
    "abstract": "ProphetBoost is a hybrid forecasting pipeline that integrates Prophet's additive decomposition with an XGBoost regressor, enhanced by an explainable-AI-driven embedded-stability feature selection strategy. Using eight years of hourly electricity consumption data from Jeju Island and meteorological observations from four weather stations, the pipeline selects a compact, stable subset of 14 predictors from 111 candidates, and achieves a 16.77% improvement in test MAE (from 6.44 MW to 5.36 MW) compared with the baseline without feature selection.",
}

# Published-paper reference numbers (from the paper's own tables)
N_CANDIDATE_FEATS_REF = 111
N_SELECTED_FEATS_REF = 14
MAE_WITHOUT_FS_REF = 6.44
MAE_WITH_FS_REF = 5.36
MAE_IMPROVEMENT_PCT = 16.77

ACCURACY_WITH_FS = {
    "ProphetBoost": dict(MAE=5.36, RMSE=9.25, MAPE=0.77),
    "TFT": dict(MAE=8.92, RMSE=15.76, MAPE=1.37),
    "Autoformer": dict(MAE=9.61, RMSE=16.47, MAPE=1.44),
    "DeepAR": dict(MAE=10.14, RMSE=17.14, MAPE=1.53),
    "Transformer": dict(MAE=10.44, RMSE=17.47, MAPE=1.56),
    "CNN-LSTM": dict(MAE=10.80, RMSE=18.14, MAPE=1.59),
    "Deep RVFL": dict(MAE=14.86, RMSE=23.04, MAPE=2.24),
    "DLinear": dict(MAE=16.85, RMSE=25.93, MAPE=2.52),
    "CNN-ANN": dict(MAE=23.62, RMSE=32.46, MAPE=3.57),
    "ARIMA-LSTM": dict(MAE=124.77, RMSE=145.10, MAPE=20.78),
}

SELECTED_FEATURES_REF = [
    ("roll3_mean", 45.3), ("lag_1", 9.4), ("JEJU_DI", 2.4), ("SEOGWIPO_DI", 2.3),
    ("hour", 2.0), ("GOSAN_DI", 1.2), ("GOSAN_SI_lag1", 1.1), ("roll3_std", 1.0),
    ("GOSAN_SI", 0.4), ("JEJU_SI_lag1", 0.4), ("SEOGWIPO_DI_lag1", 0.3),
    ("SUNGSAN_DI", 0.3), ("GOSAN_TD", 0.2), ("SUNGSAN_TA", 0.1),
]

WEATHER_PREFIXES = ["JEJU_", "GOSAN_", "SUNGSAN_", "SEOGWIPO_"]

# --------------------------------------------------------------------------
# i18n — short UI strings only. The paper's own English text stays as-is.
# --------------------------------------------------------------------------
I18N = {
    "EN": {
        "brand_sub": "Artificial Intelligence Lab, Jeju National University",
        "badge_live": "Live",
        "nav_Overview": "Overview", "nav_Forecast": "Forecast", "nav_Data": "Data", "nav_About": "About",

        "ov_eyebrow": "Electricity load forecasting",
        "ov_title": "Forecast electricity demand in one click",
        "ov_lead": "Pick a dataset, pick your models, and see how well they predict — with live charts and side-by-side scores.",
        "ov_start": "Start Forecasting →",
        "p1_t": "Choose your data", "p1_d": "Use the built-in Jeju Island dataset, or upload your own CSV.",
        "p2_t": "Pick models", "p2_d": "XGBoost, LightGBM, Random Forest, neural networks and more — run several at once.",
        "p3_t": "See the forecast", "p3_d": "Watch predictions unfold live, then compare accuracy across models.",
        "ov_models_title": "Models you can run",
        "ov_bench_title": "What's possible on this data",
        "ov_bench_cap": "Published accuracy on the Jeju dataset — lower MAE is better.",

        "fc_eyebrow": "Forecast", "fc_title": "Run a forecast",
        "fc_sub": "Three quick choices, then press Run.",
        "fc_step1": "1 · Data", "fc_step2": "2 · Models", "fc_step3": "3 · Settings",
        "fc_dataset": "Dataset", "fc_target": "What to predict", "fc_models": "Models to run",
        "fc_testsize": "Hold out the last", "fc_rows": "Rows to use",
        "fc_run": "▶ Run Forecast", "fc_running": "Forecasting…",
        "fc_best": "Best model", "fc_compare": "Model comparison",
        "fc_live": "Live forecast", "fc_replay": "▶ Replay animation",
        "fc_importance": "What drove the forecast",
        "fc_empty": "Pick your options above and press Run Forecast.",
        "fc_download": "Download forecast (CSV)",

        "d_eyebrow": "Data", "d_title": "Your datasets",
        "d_sub": "Browse the built-in dataset or add your own CSV.",
        "tab_builtin": "Jeju electricity load", "tab_yours": "Your uploads", "tab_upload": "Upload CSV",
        "m_rows": "Rows", "m_cols": "Columns", "m_range": "Period", "m_load": "Load range (MW)",
        "d_chart": "Monthly average load (MW)", "d_preview": "Preview data", "d_download": "Download CSV",

        "a_eyebrow": "About", "a_title": "The research behind this",
        "a_sub": "This dashboard runs the ProphetBoost pipeline from our published paper.",
        "a_paper": "The paper", "a_figs": "Figures", "a_bench": "Full benchmark",
    },
    "KO": {
        "brand_sub": "인공지능 연구실, 제주대학교",
        "badge_live": "실시간",
        "nav_Overview": "개요", "nav_Forecast": "예측", "nav_Data": "데이터", "nav_About": "소개",

        "ov_eyebrow": "전력 수요 예측",
        "ov_title": "클릭 한 번으로 전력 수요를 예측하세요",
        "ov_lead": "데이터와 모델을 고르면, 예측 정확도를 실시간 차트와 비교 점수로 보여드립니다.",
        "ov_start": "예측 시작하기 →",
        "p1_t": "데이터 선택", "p1_d": "내장된 제주도 데이터를 쓰거나, 직접 CSV를 올리세요.",
        "p2_t": "모델 선택", "p2_d": "XGBoost, LightGBM, 랜덤 포레스트, 신경망 등 여러 모델을 한 번에 실행합니다.",
        "p3_t": "예측 확인", "p3_d": "예측이 펼쳐지는 과정을 보고, 모델별 정확도를 비교하세요.",
        "ov_models_title": "사용할 수 있는 모델",
        "ov_bench_title": "이 데이터로 가능한 성능",
        "ov_bench_cap": "제주 데이터에 대한 논문 결과 — MAE가 낮을수록 좋습니다.",

        "fc_eyebrow": "예측", "fc_title": "예측 실행하기",
        "fc_sub": "세 가지만 고르고 실행을 누르세요.",
        "fc_step1": "1 · 데이터", "fc_step2": "2 · 모델", "fc_step3": "3 · 설정",
        "fc_dataset": "데이터셋", "fc_target": "예측할 값", "fc_models": "실행할 모델",
        "fc_testsize": "마지막 구간 평가 비율", "fc_rows": "사용할 행 수",
        "fc_run": "▶ 예측 실행", "fc_running": "예측 중…",
        "fc_best": "최고 성능 모델", "fc_compare": "모델 비교",
        "fc_live": "실시간 예측", "fc_replay": "▶ 애니메이션 다시 보기",
        "fc_importance": "예측에 영향을 준 요인",
        "fc_empty": "위에서 옵션을 고르고 예측 실행을 누르세요.",
        "fc_download": "예측 결과 내려받기 (CSV)",

        "d_eyebrow": "데이터", "d_title": "데이터셋",
        "d_sub": "내장 데이터를 살펴보거나 직접 CSV를 추가하세요.",
        "tab_builtin": "제주 전력 수요", "tab_yours": "내 업로드", "tab_upload": "CSV 업로드",
        "m_rows": "행", "m_cols": "열", "m_range": "기간", "m_load": "수요 범위 (MW)",
        "d_chart": "월별 평균 수요 (MW)", "d_preview": "데이터 미리보기", "d_download": "CSV 내려받기",

        "a_eyebrow": "소개", "a_title": "이 대시보드의 연구 배경",
        "a_sub": "이 대시보드는 발표된 논문의 ProphetBoost 파이프라인을 실행합니다.",
        "a_paper": "논문", "a_figs": "그림", "a_bench": "전체 벤치마크",
    },
}


def t(key):
    lang = st.session_state.get("lang", "EN")
    return I18N.get(lang, I18N["EN"]).get(key, I18N["EN"].get(key, key))


st.set_page_config(page_title="ProphetBoost Studio", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_load_df():
    df = pd.read_csv(LOAD_CSV, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["BASE_DT"]).copy()
    dt1 = pd.to_datetime(df["BASE_DT"], format="%d-%m-%y %H:%M", errors="coerce")
    mask = dt1.isna()
    if mask.any():
        dt2 = pd.to_datetime(df.loc[mask, "BASE_DT"], format="%m/%d/%Y %H:%M", errors="coerce")
        dt1.loc[mask] = dt2
    df["datetime"] = dt1
    return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_uploaded_df(stored_filename: str):
    path = UPLOAD_DIR / stored_filename
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def read_manifest():
    if not MANIFEST_PATH.exists():
        return []
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def write_manifest(records):
    MANIFEST_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def dataset_registry():
    reg = {"Jeju electricity load (hourly)": {"kind": "load"}}
    for rec in read_manifest():
        reg[rec["name"]] = {"kind": "uploaded", "record": rec}
    return reg


def get_df_for(entry):
    if entry["kind"] == "load":
        return load_load_df(), "datetime"
    df = load_uploaded_df(entry["record"]["stored_filename"])
    date_col = None
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) and any(k in c.lower() for k in ("date", "time", "_dt")):
            date_col = c
            break
    return df, date_col


def delete_uploaded(rec_id):
    records = read_manifest()
    for r in [r for r in records if r["id"] == rec_id]:
        p = UPLOAD_DIR / r["stored_filename"]
        if p.exists():
            p.unlink()
    write_manifest([r for r in records if r["id"] != rec_id])
    load_uploaded_df.clear()


def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


# --------------------------------------------------------------------------
# Feature engineering (fast — no Prophet, so models run in seconds)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame, target: str, date_col: str | None):
    """Calendar + lag + rolling features, plus any other numeric columns."""
    work = df.copy()
    if date_col and date_col in work.columns:
        dt = pd.to_datetime(work[date_col], errors="coerce")
        work["hour"] = dt.dt.hour
        work["day_of_week"] = dt.dt.dayofweek
        work["month"] = dt.dt.month
        work["is_weekend"] = (work["day_of_week"] >= 5).astype(int)

    y = work[target]
    work["lag_1"] = y.shift(1)
    work["lag_24"] = y.shift(24)
    work["lag_168"] = y.shift(168)
    work["roll3_mean"] = y.rolling(3).mean()
    work["roll3_std"] = y.rolling(3).std()
    work["roll24_mean"] = y.rolling(24).mean()

    exog = [c for c in numeric_columns(df) if c != target]
    for c in exog:
        work[f"{c}_lag1"] = work[c].shift(1)

    feats = [c for c in numeric_columns(work) if c != target]
    work = work[feats + [target] + ([date_col] if date_col else [])].dropna().reset_index(drop=True)
    return work, feats


MODEL_BUILDERS = {
    "XGBoost": lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                        n_jobs=-1, random_state=42),
    "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                          n_jobs=-1, random_state=42, verbose=-1),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=150, max_depth=14, n_jobs=-1,
                                                   random_state=42),
    "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=150, max_depth=16, n_jobs=-1,
                                               random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                           learning_rate=0.05, random_state=42),
    "Neural Network": lambda: make_pipeline(StandardScaler(),
                                            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=220,
                                                         early_stopping=True, random_state=42)),
    "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "K-Nearest Neighbors": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, n_jobs=-1)),
}

MODEL_NOTE = {
    "XGBoost": "Gradient-boosted trees. Fast and usually the strongest on tabular data.",
    "LightGBM": "Leaf-wise gradient boosting. Very fast on large datasets.",
    "Random Forest": "Many independent trees, averaged. Robust, little tuning needed.",
    "Extra Trees": "Like Random Forest with more randomised splits.",
    "Gradient Boosting": "Classic sequential boosting. Slower but steady.",
    "Neural Network": "A multi-layer perceptron (deep-learning style, CPU-friendly).",
    "Ridge Regression": "Linear model with regularisation. A solid baseline.",
    "Linear Regression": "The simplest baseline — a straight-line fit.",
    "K-Nearest Neighbors": "Predicts from the most similar past hours.",
    "ProphetBoost": "Our paper's hybrid: Prophet seasonality + stability-selected XGBoost.",
}


def metrics_of(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    nz = np.asarray(y_true) != 0
    mape = float(np.mean(np.abs((np.asarray(y_true)[nz] - np.asarray(y_pred)[nz]) / np.asarray(y_true)[nz])) * 100) if nz.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# --------------------------------------------------------------------------
# ProphetBoost pipeline (the paper's method — kept intact)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_engineered_load_df():
    df = load_load_df().copy().rename(columns={"datetime": "ds"})
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])
    df["y"] = df["TOTAL_LOAD"]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    m.fit(df[["ds", "y"]])
    forecast = m.predict(df[["ds", "y"]])
    df["trend"] = forecast["trend"].values
    df["seasonality"] = forecast["yearly"].values

    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["lag_1"] = df["y"].shift(1)
    df["lag_24"] = df["y"].shift(24)
    df["roll3_mean"] = df["y"].rolling(3).mean()
    df["roll3_std"] = df["y"].rolling(3).std()

    weather_feats = [c for c in df.columns if any(c.startswith(p) for p in WEATHER_PREFIXES)]
    for feat in weather_feats:
        df[f"{feat}_lag1"] = df[feat].shift(1)
        df[f"{feat}_lag24"] = df[feat].shift(24)

    df = df.dropna().reset_index(drop=True)
    base = ["trend", "seasonality", "hour", "day_of_week", "is_weekend", "lag_1", "lag_24",
            "roll3_mean", "roll3_std"]
    lagged = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_lag24")]
    return df, base + weather_feats + lagged


def run_prophetboost(split_frac, B=15, K=15, tau=0.6, progress_cb=None):
    engineered_df, full_feats = build_engineered_load_df()
    split = int(len(engineered_df) * (1 - split_frac))
    train = engineered_df.iloc[:split]
    test = engineered_df.iloc[split:]

    vt = VarianceThreshold(threshold=1e-8).fit(train[full_feats])
    filtered = [f for f, keep in zip(full_feats, vt.get_support()) if keep]

    counts = Counter()
    for b in range(B):
        sample = train.sample(frac=1.0, replace=True, random_state=b)
        model_b = xgb.train(
            {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": 4,
             "learning_rate": 0.1, "seed": b},
            xgb.DMatrix(sample[filtered], label=sample["y"]), num_boost_round=50)
        imp = model_b.get_score(importance_type="gain")
        counts.update([f for f, _ in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:K]])
        if progress_cb:
            progress_cb(b + 1, B)

    selected = [f for f, c in counts.items() if c >= tau * B] or [f for f, _ in counts.most_common(8)]

    params = {"objective": "reg:squarederror", "tree_method": "hist", "learning_rate": 0.05,
              "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "mae", "seed": 42}
    dtrain = xgb.DMatrix(train[selected], label=train["y"])
    cv = xgb.cv(params, dtrain, num_boost_round=400, nfold=5, early_stopping_rounds=20,
                metrics="mae", as_pandas=True, seed=42)
    bst = xgb.train(params, dtrain, num_boost_round=len(cv))

    y_pred = bst.predict(xgb.DMatrix(test[selected], label=test["y"]))
    gain = bst.get_score(importance_type="gain")
    return {
        "y_true": test["y"].values, "y_pred": y_pred,
        "dates": test["ds"].astype(str).tolist(),
        "importances": {f: gain.get(f, 0.0) for f in selected},
        "n_selected": len(selected), "n_candidates": len(full_feats),
        "n_train": len(train), "n_test": len(test),
    }


# --------------------------------------------------------------------------
# Theming
# --------------------------------------------------------------------------
def inject_css(dark: bool):
    if dark:
        bg, surface, surface2 = "#0c0e1a", "#12162a", "#181d35"
        text, text2, muted = "#eef0f8", "#b7bdd0", "#7d84a0"
        border = "rgba(255,255,255,0.10)"
        accent, accent_soft, accent_ink = "#7c8cf0", "#232a5c", "#c3caf7"
        good_bg, good_ink = "#123a22", "#4ade80"
    else:
        bg, surface, surface2 = "#f4f6fb", "#ffffff", "#eef1f8"
        text, text2, muted = "#12172e", "#5b6478", "#8890a0"
        border = "#e3e7f0"
        accent, accent_soft, accent_ink = "#3b4fd6", "#e4e7fc", "#2a3aa8"
        good_bg, good_ink = "#e5f6ea", "#1f8a4c"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"] {{ font-family:'IBM Plex Sans',-apple-system,'Segoe UI',sans-serif; }}
    h1,h2,h3,h4 {{ font-family:'Sora',sans-serif !important; font-weight:700 !important; letter-spacing:-.01em; color:{text}; }}
    code, .mono, [data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace !important; }}

    #MainMenu, footer {{ visibility:hidden; }}
    [data-testid="stHeader"] {{ display:none; }}
    [data-testid="collapsedControl"], section[data-testid="stSidebar"] {{ display:none; }}
    [data-testid="stAppViewContainer"] {{ background:{bg}; }}
    .stMarkdown, p, li, label, span {{ color:{text}; }}
    .block-container {{ padding-top:1.2rem; max-width:1150px; }}

    .brand {{ display:flex; align-items:center; gap:10px; }}
    .brand .logo {{ width:38px;height:38px;border-radius:11px;background:linear-gradient(155deg,{accent},{accent_ink});
                    display:flex;align-items:center;justify-content:center;font-size:18px;flex:none; }}
    .brand .t1 {{ font-family:'Sora',sans-serif;font-weight:700;font-size:15.5px;line-height:1.15;color:{text}; }}
    .brand .t2 {{ font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:{muted};line-height:1.3; }}
    .badge {{ display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;
              padding:6px 12px;border-radius:100px;background:{good_bg};color:{good_ink};white-space:nowrap; }}
    .badge-dot {{ width:6px;height:6px;border-radius:50%;background:currentColor;display:inline-block; }}
    div[data-testid="stButton"] button {{ border-radius:100px !important; font-weight:600 !important;
        font-size:13.5px !important; padding:0.3rem 0.85rem !important; min-width:0 !important; }}
    div[data-testid="stButton"] button[kind="secondary"] {{ border-color:{border} !important; color:{text2} !important; background:{surface} !important; }}
    div[data-testid="stButton"] button[kind="primary"] {{ background:{accent_soft} !important; color:{accent_ink} !important; border-color:{accent_soft} !important; box-shadow:none !important; }}
    hr {{ border-color:{border}; margin:8px 0 22px; }}

    .eyebrow {{ font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{accent};margin-bottom:6px; }}
    .pagetitle {{ font-family:'Sora',sans-serif;font-weight:800;font-size:clamp(24px,3vw,32px);color:{text};letter-spacing:-.01em; }}
    .pagesub {{ color:{text2};font-size:14.5px;margin-top:6px;max-width:70ch; }}
    .hero {{ background:linear-gradient(135deg,{surface} 0%,{surface2} 100%);border:1px solid {border};
             border-radius:22px;padding:40px 44px;margin-bottom:24px; }}
    .card {{ background:{surface};border:1px solid {border};border-radius:16px;padding:20px 22px;margin-bottom:14px; }}
    .muted {{ color:{muted};font-size:12.5px; }}
    .step {{ display:flex;gap:14px;align-items:flex-start;padding:16px 18px;background:{surface};
             border:1px solid {border};border-radius:14px;height:100%; }}
    .step .circ {{ width:30px;height:30px;border-radius:50%;background:{accent_soft};color:{accent_ink};
                   display:flex;align-items:center;justify-content:center;font-family:'IBM Plex Mono',monospace;
                   font-weight:700;font-size:13px;flex:none; }}
    .step .t {{ font-size:14px;font-weight:700;color:{text}; }}
    .step .d {{ font-size:12.5px;color:{muted};margin-top:2px; }}
    .mchip {{ display:inline-block;background:{surface};border:1px solid {border};border-radius:100px;
              padding:7px 14px;margin:0 6px 8px 0;font-size:12.5px;font-weight:600;color:{text2}; }}
    [data-testid="stMetric"] {{ background:{surface};border:1px solid {border};border-radius:14px;padding:14px 18px; }}
    [data-testid="stMetricLabel"] p {{ color:{muted} !important;font-size:10.5px !important;font-weight:700 !important;
        letter-spacing:.08em !important;text-transform:uppercase !important; }}
    [data-testid="stMetricValue"] {{ color:{text} !important;font-size:22px !important;white-space:nowrap;
        overflow:hidden;text-overflow:ellipsis; }}
    .stTabs [aria-selected="true"] {{ color:{accent} !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color:{accent} !important; }}
    [data-testid="stFileUploaderDropzone"] {{ background:{surface2};border:1.5px dashed {border};border-radius:14px; }}
    [data-testid="stExpander"] {{ background:{surface};border:1px solid {border};border-radius:14px; }}
    .footerbar {{ display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;padding:22px 0 8px;
                  margin-top:30px;border-top:1px solid {border};font-size:11.5px;color:{muted}; }}
    </style>
    """, unsafe_allow_html=True)


def plotly_theme(dark: bool):
    text = "#eef0f8" if dark else "#12172e"
    muted = "#7d84a0" if dark else "#8890a0"
    grid = "#232a44" if dark else "#e9ecf4"
    paper = "#12162a" if dark else "#ffffff"
    return dict(
        paper_bgcolor=paper, plot_bgcolor=paper,
        font=dict(color=text, family="IBM Plex Sans, sans-serif", size=12),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=muted)),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=muted)),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor=text, font_color=paper, font_family="IBM Plex Mono, monospace"),
    )


# --------------------------------------------------------------------------
# Top navigation
# --------------------------------------------------------------------------
NAV_ITEMS = ["Overview", "Forecast", "Data", "About"]
NAV_RATIOS = {
    "EN": [3.0, 1.0, 0.95, 0.8, 0.8, 0.68, 0.6, 0.6, 0.42],
    "KO": [3.0, 0.78, 0.78, 0.82, 0.78, 0.78, 0.6, 0.6, 0.42],
}

st.session_state.setdefault("page", "Overview")
st.session_state.setdefault("dark_mode", False)
st.session_state.setdefault("lang", "EN")

inject_css(st.session_state.dark_mode)
PT = plotly_theme(st.session_state.dark_mode)

top = st.columns(NAV_RATIOS.get(st.session_state.lang, NAV_RATIOS["EN"]))
with top[0]:
    st.markdown(
        f'<div class="brand"><div class="logo">⚡</div><div><div class="t1">ProphetBoost Studio</div>'
        f'<div class="t2">{t("brand_sub")}</div></div></div>', unsafe_allow_html=True)
for i, item in enumerate(NAV_ITEMS):
    with top[i + 1]:
        if st.button(t(f"nav_{item}"), key=f"nav_{item}",
                     type="primary" if st.session_state.page == item else "secondary",
                     use_container_width=True):
            st.session_state.page = item
            st.rerun()
with top[5]:
    st.markdown(f'<div style="padding-top:6px;"><span class="badge"><span class="badge-dot"></span>{t("badge_live")}</span></div>',
                unsafe_allow_html=True)
with top[6]:
    if st.button("EN", key="lang_en", type="primary" if st.session_state.lang == "EN" else "secondary",
                 use_container_width=True):
        st.session_state.lang = "EN"
        st.rerun()
with top[7]:
    if st.button("KO", key="lang_ko", type="primary" if st.session_state.lang == "KO" else "secondary",
                 use_container_width=True):
        st.session_state.lang = "KO"
        st.rerun()
with top[8]:
    st.toggle("🌙", key="dark_mode", label_visibility="collapsed")
st.markdown("<hr>", unsafe_allow_html=True)


def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="pagetitle">{title}</div>'
                f'<div class="pagesub">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def goto(page_name):
    st.session_state.page = page_name
    st.rerun()


def render_footer():
    st.markdown(f'<div class="footerbar"><span>ProphetBoost Studio</span>'
                f'<span>Artificial Intelligence Lab · Jeju National University</span></div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def render_pdf_figure(path_str: str, dpi: int = 190):
    doc = pymupdf.open(path_str)
    return doc[0].get_pixmap(dpi=dpi).tobytes("png")


def show_figure(path: Path, caption: str):
    if not path.exists():
        return
    if path.suffix.lower() == ".pdf":
        st.image(render_pdf_figure(str(path)), caption=caption, use_container_width=True)
    else:
        st.image(str(path), caption=caption, use_container_width=True)


def bar_chart_h(labels, values, colors, title=""):
    order = np.argsort(values)
    fig = go.Figure(go.Bar(
        x=[values[i] for i in order], y=[labels[i] for i in order], orientation="h",
        marker_color=[colors[i % len(colors)] for i in order],
        text=[f"{values[i]:,.2f}" if abs(values[i]) < 100 else f"{values[i]:,.0f}" for i in order],
        textposition="outside"))
    fig.update_layout(**PT, height=max(220, 34 * len(labels) + 40), title=title, showlegend=False)
    return fig


# ==========================================================================
# PAGE: Overview
# ==========================================================================
def page_overview():
    st.markdown(f"""<div class="hero">
        <div class="eyebrow">{t("ov_eyebrow")}</div>
        <div class="pagetitle" style="font-size:clamp(26px,3.4vw,36px);">{t("ov_title")}</div>
        <p class="pagesub" style="font-size:16px;">{t("ov_lead")}</p>
        </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (n, key) in zip([c1, c2, c3], [("1", "p1"), ("2", "p2"), ("3", "p3")]):
        col.markdown(f'<div class="step"><div class="circ">{n}</div><div>'
                     f'<div class="t">{t(f"{key}_t")}</div><div class="d">{t(f"{key}_d")}</div>'
                     f'</div></div>', unsafe_allow_html=True)

    st.write("")
    b1, b2, b3 = st.columns([1, 1, 1])
    with b2:
        if st.button(t("ov_start"), key="cta_forecast", type="primary", use_container_width=True):
            goto("Forecast")

    st.write("")
    st.markdown(f"#### {t('ov_models_title')}")
    chips = "".join(f'<span class="mchip">{m}</span>' for m in ["ProphetBoost"] + list(MODEL_BUILDERS))
    st.markdown(chips, unsafe_allow_html=True)

    st.write("")
    st.markdown(f"#### {t('ov_bench_title')}")
    st.caption(t("ov_bench_cap"))
    models = list(ACCURACY_WITH_FS)
    st.plotly_chart(bar_chart_h(models, [ACCURACY_WITH_FS[m]["MAE"] for m in models], CATEGORICAL,
                                "Test MAE (MW)"),
                    use_container_width=True, config={"displayModeBar": False})


# ==========================================================================
# PAGE: Forecast
# ==========================================================================
def play_animation(chart_ph, metric_ph, y_true, y_pred, label, unit, autoplay: bool):
    """Animated reveal of the forecast, drawn into pre-reserved placeholders so
    the rest of the results page renders immediately instead of waiting."""
    window = int(min(160, len(y_true)))
    yt, yp = np.asarray(y_true)[-window:], np.asarray(y_pred)[-window:]

    def frame(i):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(i)), y=yt[:i], name="Actual", mode="lines",
                                 line=dict(color="#8890a0", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=list(range(i)), y=yp[:i], name=label, mode="lines",
                                 line=dict(color=SERIES_BLUE, width=2.5)))
        fig.update_layout(**PT, height=360, xaxis_title="Step", yaxis_title=unit,
                          xaxis_range=[0, window], showlegend=True,
                          legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"))
        chart_ph.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with metric_ph.container():
            m1, m2 = st.columns(2)
            m1.metric("Revealed", f"{i} / {window}")
            m2.metric("Running MAE", f"{float(np.mean(np.abs(yt[:i] - yp[:i]))):,.2f}")

    if autoplay:
        for i in range(2, window + 1, max(1, window // 45)):
            frame(i)
            time.sleep(0.05)
    frame(window)


def page_forecast():
    page_header(t("fc_eyebrow"), t("fc_title"), t("fc_sub"))

    reg = dataset_registry()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**{t('fc_step1')}**")
        label = st.selectbox(t("fc_dataset"), list(reg), key="fc_ds")
        entry = reg[label]
        df, date_col = get_df_for(entry)
        num_cols = numeric_columns(df)
        if not num_cols:
            st.warning("This dataset has no numeric columns to forecast.")
            return
        default_target = "TOTAL_LOAD" if "TOTAL_LOAD" in num_cols else num_cols[-1]
        target = st.selectbox(t("fc_target"), num_cols,
                              index=num_cols.index(default_target), key="fc_target")
    with c2:
        st.markdown(f"**{t('fc_step2')}**")
        options = list(MODEL_BUILDERS)
        if entry["kind"] == "load" and target == "TOTAL_LOAD":
            options = ["ProphetBoost"] + options
        picked = st.multiselect(t("fc_models"), options,
                                default=["XGBoost", "Random Forest", "LightGBM"], key="fc_models")

    with st.expander(t("fc_step3")):
        s1, s2 = st.columns(2)
        test_pct = s1.slider(t("fc_testsize"), 10, 40, 20, key="fc_test") / 100
        row_options = sorted({n for n in (2000, 5000, 10000, 20000, 40000) if n < len(df)} | {len(df)})
        max_rows = s2.select_slider(t("fc_rows"), options=row_options,
                                    value=min(20000, len(df)) if min(20000, len(df)) in row_options else row_options[-1],
                                    key="fc_rows")

    run = st.button(t("fc_run"), type="primary")
    st.write("")

    if run:
        if not picked:
            st.warning("Pick at least one model.")
            return
        results, prog = {}, st.progress(0.0, text=t("fc_running"))
        work, feats = None, None
        if [n for n in picked if n != "ProphetBoost"]:
            work, feats = build_features(df.tail(int(max_rows)), target, date_col)
            if len(work) < 50:
                st.error("Not enough complete rows after building lag features. Try a larger row count.")
                return
        for idx, name in enumerate(picked):
            prog.progress(idx / len(picked), text=f"{t('fc_running')} {name}")
            t0 = time.time()
            try:
                if name == "ProphetBoost":
                    out = run_prophetboost(test_pct)
                    res = {"y_true": out["y_true"], "y_pred": out["y_pred"],
                           "importances": out["importances"], "n_test": out["n_test"]}
                else:
                    split = int(len(work) * (1 - test_pct))
                    X_tr, X_te = work[feats].iloc[:split], work[feats].iloc[split:]
                    y_tr, y_te = work[target].iloc[:split], work[target].iloc[split:]
                    model = MODEL_BUILDERS[name]()
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_te)
                    est = model[-1] if hasattr(model, "steps") else model
                    if hasattr(est, "feature_importances_"):
                        imps = dict(zip(feats, est.feature_importances_))
                    elif hasattr(est, "coef_"):
                        imps = dict(zip(feats, np.abs(np.ravel(est.coef_))))
                    else:
                        imps = {}
                    res = {"y_true": y_te.values, "y_pred": pred, "importances": imps,
                           "n_test": len(y_te)}
                res["metrics"] = metrics_of(res["y_true"], res["y_pred"])
                res["seconds"] = time.time() - t0
                results[name] = res
            except Exception as e:
                st.error(f"{name} failed: {e}")
        prog.empty()
        if results:
            st.session_state["forecast"] = {
                "results": results, "target": target, "dataset": label,
                "unit": "MW" if target == "TOTAL_LOAD" else target, "fresh": True,
            }

    state = st.session_state.get("forecast")
    if not state:
        st.markdown(f'<div class="card" style="text-align:center;padding:40px;">'
                    f'<div style="font-size:26px;">📈</div><p class="muted">{t("fc_empty")}</p></div>',
                    unsafe_allow_html=True)
        return

    results = state["results"]
    ranked = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["MAE"])
    best_name, best = ranked[0]

    st.markdown(f"#### {t('fc_best')} · {best_name}")
    m = best["metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE", f"{m['MAE']:,.2f}")
    k2.metric("RMSE", f"{m['RMSE']:,.2f}")
    k3.metric("R²", f"{m['R2']:.3f}")
    k4.metric("MAPE", f"{m['MAPE']:.2f}%" if not np.isnan(m["MAPE"]) else "—")
    st.caption(f"{state['dataset']} · {best['n_test']:,} test points · trained in {best['seconds']:.1f}s")

    st.write("")
    st.markdown(f"#### {t('fc_live')}")
    autoplay = state.pop("fresh", False) or st.button(t("fc_replay"))
    anim_chart_ph, anim_metric_ph = st.empty(), st.empty()

    if len(results) > 1:
        st.write("")
        st.markdown(f"#### {t('fc_compare')}")
        names = [n for n, _ in ranked]
        st.plotly_chart(bar_chart_h(names, [results[n]["metrics"]["MAE"] for n in names],
                                    CATEGORICAL, f"MAE ({state['unit']})"),
                        use_container_width=True, config={"displayModeBar": False})
        table = pd.DataFrame([{
            "Model": n, "MAE": round(results[n]["metrics"]["MAE"], 3),
            "RMSE": round(results[n]["metrics"]["RMSE"], 3),
            "R²": round(results[n]["metrics"]["R2"], 4),
            "MAPE (%)": round(results[n]["metrics"]["MAPE"], 2),
            "Seconds": round(results[n]["seconds"], 1)} for n in names])
        st.dataframe(table, use_container_width=True, hide_index=True)

    if best["importances"]:
        st.write("")
        st.markdown(f"#### {t('fc_importance')}")
        imp = sorted(best["importances"].items(), key=lambda kv: kv[1], reverse=True)[:10]
        st.plotly_chart(bar_chart_h([k for k, _ in imp], [float(v) for _, v in imp], CATEGORICAL),
                        use_container_width=True, config={"displayModeBar": False})

    out_df = pd.DataFrame({"actual": best["y_true"], "predicted": best["y_pred"]})
    out_df["error"] = out_df["actual"] - out_df["predicted"]
    st.download_button(t("fc_download"), out_df.to_csv(index=False).encode(),
                       file_name=f"forecast_{best_name.lower().replace(' ', '_')}.csv", mime="text/csv")

    # Played last so the whole results page is already on screen while it runs.
    play_animation(anim_chart_ph, anim_metric_ph, best["y_true"], best["y_pred"],
                   best_name, state["unit"], autoplay)


# ==========================================================================
# PAGE: Data
# ==========================================================================
def page_data():
    page_header(t("d_eyebrow"), t("d_title"), t("d_sub"))
    tab1, tab2, tab3 = st.tabs([t("tab_builtin"), t("tab_yours"), t("tab_upload")])

    with tab1:
        df = load_load_df()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("m_rows"), f"{len(df):,}")
        c2.metric(t("m_cols"), len(df.columns))
        c3.metric(t("m_range"), f"{df['datetime'].min():%Y} → {df['datetime'].max():%Y}")
        c4.metric(t("m_load"), f"{df['TOTAL_LOAD'].min():.0f}–{df['TOTAL_LOAD'].max():.0f}")

        monthly = df.set_index("datetime")["TOTAL_LOAD"].resample("MS").mean()
        fig = go.Figure(go.Scatter(x=monthly.index, y=monthly.values, mode="lines",
                                   line=dict(color=SERIES_BLUE, width=2), fill="tozeroy",
                                   fillcolor="rgba(42,120,214,0.12)"))
        fig.update_layout(**PT, height=300, yaxis_title="MW", showlegend=False)
        st.markdown(f"**{t('d_chart')}**")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander(t("d_preview")):
            st.dataframe(df.head(50), use_container_width=True)
        st.download_button(t("d_download"), LOAD_CSV.read_bytes(), file_name=LOAD_CSV.name,
                           mime="text/csv")

    with tab2:
        records = read_manifest()
        if not records:
            st.info("No uploads yet — add a CSV in the next tab.")
        for rec in records:
            with st.container(border=True):
                top_r = st.columns([4, 1])
                top_r[0].markdown(f"**{rec['name']}**")
                top_r[0].caption(f"{rec['original_filename']} · {rec['rows']:,} rows · {rec['columns']} columns")
                if top_r[1].button("Remove", key=f"del_{rec['id']}"):
                    delete_uploaded(rec["id"])
                    st.rerun()
                try:
                    st.dataframe(load_uploaded_df(rec["stored_filename"]).head(8),
                                 use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read this file: {e}")

    with tab3:
        file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if file is not None:
            try:
                df = pd.read_csv(file)
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin-1")
            st.success(f"{len(df):,} rows × {len(df.columns)} columns")
            st.dataframe(df.head(8), use_container_width=True)
            name = st.text_input("Name it", value=Path(file.name).stem)
            if st.button("Add dataset", type="primary"):
                stored = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}_{file.name}"
                file.seek(0)
                (UPLOAD_DIR / stored).write_bytes(file.read())
                records = read_manifest()
                records.append({"id": uuid.uuid4().hex, "name": name or file.name,
                                "original_filename": file.name, "stored_filename": stored,
                                "uploaded_at": datetime.now().isoformat(),
                                "rows": len(df), "columns": len(df.columns)})
                write_manifest(records)
                st.success(f"Added — it's now selectable on the Forecast page.")


# ==========================================================================
# PAGE: About
# ==========================================================================
def page_about():
    page_header(t("a_eyebrow"), t("a_title"), t("a_sub"))

    st.markdown(f"""<div class="card">
        <h3 style="margin-bottom:8px;">{PAPER['title']}</h3>
        <p class="muted">{PAPER['authors']}</p>
        <p class="muted">{PAPER['affiliation']} · {PAPER['venue']}</p>
        <p class="muted"><a href="https://doi.org/{PAPER['doi']}" target="_blank">doi.org/{PAPER['doi']}</a></p>
        </div>""", unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Features kept", f"{N_SELECTED_FEATS_REF} / {N_CANDIDATE_FEATS_REF}")
    k2.metric("Published MAE", f"{MAE_WITH_FS_REF} MW")
    k3.metric("Improvement", f"{MAE_IMPROVEMENT_PCT}%")

    st.write("")
    st.write(PAPER["abstract"])

    with st.expander(t("a_figs")):
        show_figure(FIG_PIPELINE, "The ProphetBoost pipeline")
        show_figure(FIG_FEATURE_SELECTION, "Embedded + stability feature selection")
        show_figure(FIG_SELECTION_OUTCOME, "14 of 111 features kept")
        show_figure(FIG_MODEL_DIAGNOSTICS, "Model diagnostics")
        show_figure(FIG_SHAP_BEESWARM, "SHAP summary")

    with st.expander(t("a_bench")):
        st.caption("Published comparison against nine baselines (paper Table 9, with feature selection).")
        st.dataframe(pd.DataFrame(ACCURACY_WITH_FS).T.reset_index().rename(
            columns={"index": "Model", "MAE": "MAE (MW)", "RMSE": "RMSE (MW)", "MAPE": "MAPE (%)"}),
            use_container_width=True, hide_index=True)
        st.caption("The 14 selected predictors and their share of total gain:")
        st.dataframe(pd.DataFrame(SELECTED_FEATURES_REF, columns=["Feature", "% of gain"]),
                     use_container_width=True, hide_index=True)


ROUTES = {"Overview": page_overview, "Forecast": page_forecast, "Data": page_data, "About": page_about}
ROUTES[st.session_state.page]()
render_footer()
