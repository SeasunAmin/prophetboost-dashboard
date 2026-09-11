"""ProphetBoost Research Dashboard — local Streamlit app.

Built around the real project in this folder: totalload_new.csv (Jeju
Island hourly electricity load, 2012-2020) and mian.ipynb, which
implements ProphetBoost — Prophet trend/seasonality decomposition,
hybrid embedded+stability XGBoost feature selection, and a final
cross-validated XGBoost model. This app exposes that exact pipeline
as a live, retrainable model, plus a lighter generic trainer for any
uploaded dataset. Run with: streamlit run app.py
"""

import json
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymupdf
import streamlit as st
import xgboost as xgb
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
LOAD_CSV = BASE_DIR / "totalload_new.csv"
UPLOAD_DIR = BASE_DIR / "uploaded_datasets"
MANIFEST_PATH = UPLOAD_DIR / "manifest.json"
FEATURE_IMPORTANCE_IMG = BASE_DIR / "feature_importance.png"
WEATHER_PLOTS_IMG = BASE_DIR / "overlaid_weather_plots.png"
PAPER_PDF = BASE_DIR / "1-s2.0-S0045790626003526-main.pdf"
# Figures supplied from the published paper (matched to their real captions
# by inspecting each image, not by filename — the filenames don't line up
# with the paper's own Fig. N numbering).
FIG_PIPELINE = BASE_DIR / "fig-11.pdf"          # paper Fig. 2 — full pipeline
FIG_FEATURE_SELECTION = BASE_DIR / "fig-3.pdf"  # paper Fig. 3 — selection process
FIG_SELECTION_OUTCOME = BASE_DIR / "fig-1.png"  # paper Fig. 4 — 111 -> 14 features
FIG_MODEL_DIAGNOSTICS = BASE_DIR / "fig-2.png"  # paper Fig. 8 — learning curve etc.
FIG_SHAP_BEESWARM = BASE_DIR / "fig-7 (1).png"  # paper Fig. 10 — SHAP summary
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
CATEGORICAL = [SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA, SERIES_YELLOW, SERIES_MAGENTA, SERIES_VIOLET, SERIES_GREEN, SERIES_RED, SERIES_TEAL, SERIES_BROWN]

PAPER = {
    "title": "ProphetBoost: a hybrid pipeline for accurate and transparent electricity load forecasting",
    "authors": "Nazrul Amin, Yong-Woon Kim, Yung-Cheol Byun",
    "venue": "Computers & Electrical Engineering",
    "citation": "Computers and Electrical Engineering 138 (2026) 111280",
    "doi": "10.1016/j.compeleceng.2026.111280",
    "dates": "Received 22 October 2025 · Revised 16 April 2026 · Accepted 1 June 2026",
    "affiliation": "Dept. of Computer Engineering, Jeju National University, South Korea",
    "advisor": "Prof. Yung-Cheol Byun",
    "keywords": ["Short-term load forecasting", "Hybrid model", "Feature selection", "Explainable Artificial Intelligence (XAI)", "SHAP analysis", "Smart grids and urban energy systems"],
    "abstract": "Accurate short-term load forecasting is critical for the efficient operation of building energy systems and smart grids. However, increasing demand variability from weather fluctuations, occupant behavior, and renewable energy integration challenges conventional statistical approaches. Meanwhile, advanced machine learning models often achieve high accuracy but lack interpretability, limiting their adoption in building energy management. This study proposes ProphetBoost, a novel hybrid forecasting pipeline that integrates Prophet's additive decomposition with an XGBoost regressor, enhanced by an explainable AI (XAI)-driven embedded-stability feature selection strategy. Using eight years of hourly electricity consumption data from Jeju Island and meteorological observations from four weather stations, the pipeline generates engineered features and selects a compact, stable subset of 14 predictors from 111 candidates. Experimental results show that ProphetBoost achieves a 16.77% improvement in test Mean Absolute Error (from 6.44 MW to 5.36 MW) compared with baseline models. SHAP-based interpretability confirms that recent load dynamics and weather covariates are key drivers of building and regional demand. By combining predictive accuracy with transparency, ProphetBoost provides an effective tool for building energy management, demand response, and operational decision support in smart urban environments.",
    "conclusion": "The two-stage feature selection process achieved substantial dimensionality reduction (87.4%, from 111 to 14 features), retaining only a compact and stable subset of predictors validated through SHAP analysis. This pruning not only enhanced model interpretability but also improved forecasting accuracy, achieving a 16.77% reduction in MAE compared to the baseline without feature selection (from 6.44 MW to 5.36 MW). Comprehensive comparative experiments confirmed that ProphetBoost consistently outperformed statistical, machine learning, and deep learning baselines across all evaluation metrics.",
}

# --------------------------------------------------------------------------
# Published-paper reference tables (Tables 3, 4, 7, 8, 9, 11, 12, 13) — the
# authoritative numbers from 1-s2.0-S0045790626003526-main.pdf. Shown as
# reference results, not re-run live: the nine deep-learning/statistical
# baselines need PyTorch + Optuna tuning and take minutes to hours to train.
# --------------------------------------------------------------------------
N_CANDIDATE_FEATS_REF = 111
N_SELECTED_FEATS_REF = 14
MAE_WITHOUT_FS_REF = 6.44
MAE_WITH_FS_REF = 5.36
MAE_IMPROVEMENT_PCT = 16.77

# Table 9 — accuracy, with & without feature selection (MAE, MSE, RMSE, MAPE)
ACCURACY_WITHOUT_FS = {
    "ProphetBoost": dict(MAE=6.44, MSE=113.48, RMSE=10.65, MAPE=0.96),
    "TFT": dict(MAE=9.74, MSE=276.58, RMSE=16.63, MAPE=1.51),
    "Autoformer": dict(MAE=10.38, MSE=298.72, RMSE=17.28, MAPE=1.61),
    "DeepAR": dict(MAE=11.06, MSE=321.94, RMSE=17.94, MAPE=1.69),
    "Transformer": dict(MAE=11.50, MSE=323.37, RMSE=17.98, MAPE=1.72),
    "CNN-LSTM": dict(MAE=13.36, MSE=408.71, RMSE=20.22, MAPE=1.98),
    "Deep RVFL": dict(MAE=19.27, MSE=720.89, RMSE=26.85, MAPE=2.84),
    "CNN-ANN": dict(MAE=36.76, MSE=2346.21, RMSE=48.44, MAPE=5.54),
    "DLinear": dict(MAE=80.06, MSE=10427.29, RMSE=102.11, MAPE=12.12),
    "ARIMA-LSTM": dict(MAE=126.56, MSE=21585.01, RMSE=146.92, MAPE=21.00),
}
ACCURACY_WITH_FS = {
    "ProphetBoost": dict(MAE=5.36, MSE=85.60, RMSE=9.25, MAPE=0.77),
    "TFT": dict(MAE=8.92, MSE=248.41, RMSE=15.76, MAPE=1.37),
    "Autoformer": dict(MAE=9.61, MSE=271.35, RMSE=16.47, MAPE=1.44),
    "DeepAR": dict(MAE=10.14, MSE=293.88, RMSE=17.14, MAPE=1.53),
    "Transformer": dict(MAE=10.44, MSE=305.15, RMSE=17.47, MAPE=1.56),
    "CNN-LSTM": dict(MAE=10.80, MSE=329.16, RMSE=18.14, MAPE=1.59),
    "Deep RVFL": dict(MAE=14.86, MSE=531.01, RMSE=23.04, MAPE=2.24),
    "CNN-ANN": dict(MAE=23.62, MSE=1053.72, RMSE=32.46, MAPE=3.57),
    "DLinear": dict(MAE=16.85, MSE=672.42, RMSE=25.93, MAPE=2.52),
    "ARIMA-LSTM": dict(MAE=124.77, MSE=21055.21, RMSE=145.10, MAPE=20.78),
}
# Table 11 — computational cost, with & without feature selection
COMPLEXITY_WITHOUT_FS = {
    "ProphetBoost": dict(train_s=19.260, infer_s=0.006, infer_ms=0.0010, size_mb=2.41, gpu_mb=50.78),
    "Transformer": dict(train_s=227.692, infer_s=0.454, infer_ms=0.0795, size_mb=8.35, gpu_mb=262.61),
    "TFT": dict(train_s=270.332, infer_s=0.616, infer_ms=0.1078, size_mb=7.77, gpu_mb=183.47),
    "Autoformer": dict(train_s=308.599, infer_s=0.432, infer_ms=0.0756, size_mb=8.30, gpu_mb=307.70),
    "DeepAR": dict(train_s=201.868, infer_s=0.522, infer_ms=0.0914, size_mb=7.59, gpu_mb=193.32),
    "DLinear": dict(train_s=59.900, infer_s=0.110, infer_ms=0.0193, size_mb=4.00, gpu_mb=73.47),
    "Deep RVFL": dict(train_s=29.636, infer_s=0.065, infer_ms=0.0112, size_mb=9.95, gpu_mb=83.64),
    "CNN-LSTM": dict(train_s=175.425, infer_s=0.441, infer_ms=0.0773, size_mb=6.88, gpu_mb=179.44),
    "CNN-ANN": dict(train_s=46.586, infer_s=0.096, infer_ms=0.0168, size_mb=7.08, gpu_mb=94.67),
    "ARIMA-LSTM": dict(train_s=144.992, infer_s=0.555, infer_ms=0.0972, size_mb=5.16, gpu_mb=155.82),
}
COMPLEXITY_WITH_FS = {
    "ProphetBoost": dict(train_s=12.427, infer_s=0.008, infer_ms=0.0014, size_mb=2.36, gpu_mb=50.78),
    "Transformer": dict(train_s=248.508, infer_s=0.324, infer_ms=0.0567, size_mb=8.25, gpu_mb=261.52),
    "TFT": dict(train_s=201.252, infer_s=0.464, infer_ms=0.0812, size_mb=7.67, gpu_mb=181.32),
    "Autoformer": dict(train_s=273.298, infer_s=0.444, infer_ms=0.0778, size_mb=8.20, gpu_mb=307.61),
    "DeepAR": dict(train_s=196.911, infer_s=0.468, infer_ms=0.0819, size_mb=7.20, gpu_mb=185.85),
    "DLinear": dict(train_s=52.686, infer_s=0.067, infer_ms=0.0116, size_mb=0.48, gpu_mb=53.80),
    "Deep RVFL": dict(train_s=27.305, infer_s=0.060, infer_ms=0.0104, size_mb=9.56, gpu_mb=82.45),
    "CNN-LSTM": dict(train_s=162.603, infer_s=0.395, infer_ms=0.0691, size_mb=6.73, gpu_mb=178.10),
    "CNN-ANN": dict(train_s=40.244, infer_s=0.071, infer_ms=0.0124, size_mb=6.93, gpu_mb=93.36),
    "ARIMA-LSTM": dict(train_s=138.936, infer_s=0.539, infer_ms=0.0944, size_mb=5.16, gpu_mb=155.82),
}
# Table 12 — Wilcoxon signed-rank (vs. ProphetBoost, with FS) + Friedman test
WILCOXON_VS_PROPHETBOOST = [
    ("Transformer", 3533383, 0.00000), ("TFT", 3389021, 0.00000), ("Autoformer", 3314485, 0.00000),
    ("DeepAR", 3421157, 0.00000), ("DLinear", 1626654, 0.00000), ("Deep RVFL", 2414895, 0.00000),
    ("CNN-LSTM", 2748700, 0.00000), ("CNN-ANN", 763488, 0.00000), ("ARIMA-LSTM", 47427, 0.00000),
]
FRIEDMAN = dict(statistic=16432.5712, p=0.0, decision="Reject H0 — the models' errors differ significantly")

# Table 8 — daily / monthly error of the published ProphetBoost (with FS) run
DAILY_METRICS = [
    ("Monday", 5.46911, 76.5871, 8.75141, 0.787327), ("Tuesday", 6.08908, 114.187, 10.6858, 0.865476),
    ("Wednesday", 5.80371, 92.8051, 9.63354, 0.833226), ("Thursday", 5.83578, 101.841, 10.0916, 0.858346),
    ("Friday", 5.68422, 91.7073, 9.5764, 0.80592), ("Saturday", 5.07509, 60.4363, 7.77408, 0.74768),
    ("Sunday", 5.74737, 75.0371, 8.6624, 0.823289),
]
MONTHLY_METRICS = [
    ("January", 5.12855, 64.1707, 8.01066, 0.751638), ("February", 5.36753, 71.2805, 8.44278, 0.776285),
    ("March", 4.92101, 59.9113, 7.74024, 0.70212), ("April", 5.69471, 85.2099, 9.23092, 0.807662),
    ("May", 6.22903, 134.699, 11.606, 0.916602), ("June", 5.93373, 120.598, 10.9817, 0.858456),
    ("July", 5.57686, 85.2152, 9.23121, 0.81108), ("August", 5.98962, 91.5445, 9.56789, 0.852086),
    ("September", 5.57613, 65.315, 8.08177, 0.805526), ("October", 5.4538, 82.9426, 9.10728, 0.798967),
    ("November", 5.91268, 85.3523, 9.23863, 0.839741), ("December", 6.32106, 106.149, 10.3029, 0.891278),
]

# Table 13 — the 14 predictors actually selected in the published run
SELECTED_FEATURES_REF = [
    ("roll3_mean", 192169.75, 45.3, 0.792, 64.351, 1.00),
    ("lag_1", 39648.81, 9.4, 0.790, 10.898, 1.00),
    ("JEJU_DI", 10205.36, 2.4, -0.203, 0.871, 0.60),
    ("SEOGWIPO_DI", 9879.43, 2.3, -0.209, 3.128, 0.90),
    ("hour", 8506.18, 2.0, 0.413, 10.285, 1.00),
    ("GOSAN_DI", 5093.99, 1.2, -0.207, 0.763, 0.97),
    ("GOSAN_SI_lag1", 4523.03, 1.1, -0.053, 0.651, 1.00),
    ("roll3_std", 4056.77, 1.0, 0.047, 4.159, 1.00),
    ("GOSAN_SI", 1811.57, 0.4, 0.000, 1.174, 0.63),
    ("JEJU_SI_lag1", 1523.70, 0.4, -0.035, 0.761, 0.93),
    ("SEOGWIPO_DI_lag1", 1256.33, 0.3, -0.202, 2.814, 0.60),
    ("SUNGSAN_DI", 1097.88, 0.3, -0.218, 0.455, 0.77),
    ("GOSAN_TD", 969.28, 0.2, -0.237, 0.429, 0.63),
    ("SUNGSAN_TA", 550.36, 0.1, -0.231, 0.383, 0.60),
]

# Table 3 / Table 4 — published dataset statistics
DATASET_STATS_REF = dict(mean=578.63, std=108.72, min=0.0, max=1018.0, cv=18.8)
TOP_CORRELATIONS_REF = {
    "positive": [("GOSAN_PA", 0.175), ("JEJU_PA", 0.173), ("SUNGSAN_PA", 0.171), ("SEOGWIPO_PA", 0.163), ("GOSAN_WS", 0.135)],
    "negative": [("SUNGSAN_HM", -0.168), ("GOSAN_ST", -0.167), ("SUNGSAN_ST", -0.161), ("SEOGWIPO_ST", -0.155), ("SEOGWIPO_TA", -0.148)],
}
# Table 5 — published dataset partition
PARTITION_REF = [("Training", "2012-2018", 52560), ("Test", "2019-2020", 17520), ("Total", "2012-2020", 70080)]

WEATHER_PREFIXES = ["JEJU_", "GOSAN_", "SUNGSAN_", "SEOGWIPO_"]

GLOSSARY = {
    "EN": [
        ("MAE", "Mean Absolute Error — on average, how many megawatts the forecast is off by. Lower is better; it's in the same units as load (MW)."),
        ("RMSE", "Root Mean Squared Error — like MAE, but squares errors before averaging, so a few big misses raise it more than many small ones."),
        ("MAPE", "Mean Absolute Percentage Error — the average error as a percentage of actual load, so it's comparable across different demand levels."),
        ("R²", "How much of the load's variation the model explains, from 0 to 1. 0.99 means it captures almost all of the pattern."),
        ("Gain importance", "How much a feature reduced prediction error, summed across every tree split that used it — XGBoost's built-in way of ranking features."),
        ("Stability selection", "Instead of trusting one feature-importance ranking, retrain on 30 random resamples and keep only features that rank highly almost every time."),
    ],
    "KO": [
        ("MAE (평균절대오차)", "예측이 평균적으로 몇 메가와트(MW) 정도 벗어나는지를 나타냅니다. 값이 낮을수록 좋으며, 단위는 수요와 동일한 MW입니다."),
        ("RMSE (평균제곱근오차)", "MAE와 비슷하지만 평균을 내기 전에 오차를 제곱하므로, 작은 오차 여러 개보다 큰 오차 몇 개가 값을 더 크게 끌어올립니다."),
        ("MAPE (평균절대백분율오차)", "실제 수요 대비 평균 오차를 백분율로 나타내어, 서로 다른 수요 수준 간에도 비교할 수 있게 해줍니다."),
        ("R² (결정계수)", "모델이 수요 변동의 얼마나 많은 부분을 설명하는지를 0에서 1 사이 값으로 나타냅니다. 0.99라면 패턴의 거의 전부를 포착했다는 뜻입니다."),
        ("Gain importance (게인 중요도)", "어떤 특성이 사용된 모든 트리 분할에서 예측 오차를 얼마나 줄였는지의 총합 — XGBoost가 특성 중요도를 매기는 내장 방식입니다."),
        ("Stability selection (안정성 선택)", "하나의 특성 중요도 순위만 믿는 대신, 30번의 무작위 재표본으로 재학습하여 거의 매번 높은 순위를 차지하는 특성만 남깁니다."),
    ],
}

# --------------------------------------------------------------------------
# i18n — EN/KO UI strings. The paper's own English abstract/methodology text
# (Research & About) is left untranslated: it's a direct transcription of a
# published paper, and machine-translating its technical prose risks
# misrepresenting it. Everything else — navigation, headers, the Overview
# page, and shared labels — is fully translated.
# --------------------------------------------------------------------------
I18N = {
    "EN": {
        "brand_sub": "Machine Learning Lab",
        "badge_local": "Local · Live",
        "nav_Overview": "Overview", "nav_Dataset": "Dataset", "nav_Train & Simulate": "Train",
        "nav_Results & Validation": "Results", "nav_Research & About": "Research",

        "ov_eyebrow": "Overview and demonstration",
        "ov_title": "What is ProphetBoost?",
        "ov_p1": "A hybrid forecasting pipeline that decomposes electricity demand into trend and seasonality with Prophet, then lets XGBoost learn the nonlinear residual — weather, calendar effects, recent load dynamics — on a compact set of features chosen for being stable, not just accurate on one run.",
        "ov_p2": "That's more than fitting one model to the data. Because feature selection is bootstrap-validated, you can trust that the 14 predictors it keeps aren't a fluke of one train/test split — and because Prophet's trend/seasonality are explicit, every forecast is explainable in plain terms: this much is the season, this much is today's weather and recent load.",
        "m_hourly": "Hourly records", "m_feats": "Candidate → selected features", "m_pubmae": "Published MAE (with FS)", "m_datasets": "Datasets tracked",
        "how_title": "How does it work?",
        "step1_t": "Decompose", "step1_d": "Prophet extracts trend + yearly/weekly/daily seasonality from hourly load, fit on training data only.",
        "step2_t": "Engineer & select", "step2_d": "Calendar, lag, rolling stats and 4-station weather (+ lags) give 111 features; 30 bootstrap runs of XGBoost keep only the 14 that are stable.",
        "step3_t": "Train & evaluate", "step3_d": "A final XGBoost is cross-validated on the stable subset, then scored on 2019-2020 hours it never saw.",
        "arch_title": "See the architecture for yourself",
        "arch_caption": "This isn't a mockup — it's the actual pipeline diagram from the published paper, and every stage below is live and retrainable in this app.",
        "stat_pubmae": "Published MAE", "stat_selfeat": "Selected features", "stat_improve": "MAE improvement", "stat_window": "Test window",
        "arch1_t": "1 · Prophet decomposition", "arch1_d": "Trend g(t), Fourier seasonality s(t), holiday effects h(t), residual noise ε<sub>t</sub>. Only ĝ(t) and ŝ(t) become features — fit univariately, on load alone.",
        "arch2_t": "2 · Embedded + stability selection", "arch2_d": "A variance filter precedes B = 30 bootstrap resamples, each keeping its top K = 15 features by gain. Kept only if selected in ≥ τ·B = 18 of 30 runs.",
        "arch3_t": "3 · XGBoost ensemble", "arch3_d": "Stage-wise regression trees correcting prior residuals, tuned via 5-fold time-series CV with early stopping.",
        "arch_footnote": "Full equations, the selection algorithm, and the exact 14 selected features are on **Research & About → Methodology**.",
        "bench_title": "Published benchmark comparison",
        "bench_caption": "ProphetBoost vs. nine baselines, feature selection on — paper Table 9. Full tables (with/without FS, complexity, significance) under **Results & Validation**.",
        "glance_title": "At a glance",
        "feat_title": "What can you do with this dashboard?",
        "feat1_t": "Explore the dataset", "feat1_d": "Nine years of hourly Jeju Island load and weather, with the paper's own published statistics alongside.",
        "feat2_t": "Train the real pipeline", "feat2_d": "Run the actual Prophet + XGBoost pipeline live, with your own hyperparameters — not a precomputed demo.",
        "feat3_t": "Check results & baselines", "feat3_d": "Your live run's metrics and charts, plus the paper's benchmark against nine other forecasting models.",
        "feat4_t": "Read the research", "feat4_d": "The full methodology with equations, the published figures, and the exact features the paper selected.",
        "goto": "Go there →",
        "glossary_title": "\U0001F4D6 Learn the terms — six words you'll meet elsewhere in this dashboard",
        "cta_title": "Your turn", "cta_d": "Pick a cutoff date, tune the feature-selection hyperparameters, and the same pipeline trains on the real dataset in about a minute.",
        "cta_btn": "Train It Yourself →",

        "ds_eyebrow": "Dataset", "ds_title": "Jeju Island electricity load",
        "ds_sub": "Loaded live from totalload_new.csv on disk — every stat below is computed from the real file, not cached copies.",
        "tab_load": "⚡ Electricity load", "tab_uploaded": "\U0001F4C1 Uploaded datasets", "tab_upload_new": "⬆️ Upload new",
        "m_rows": "Rows", "m_cols": "Columns", "m_daterange": "Date range", "m_loadrange": "Load range (MW)",
        "monthly_load": "Monthly average load (MW)",

        "tr_eyebrow": "Train your model", "tr_title": "Train & Simulate",
        "tr_sub": "Teach the real pipeline on the real dataset, then watch its forecast unfold hour by hour on data it never saw.",
        "tab_pb": "⚡ ProphetBoost (real pipeline)", "tab_generic": "\U0001F527 Generic model (any dataset)",
        "step1_exp": "① Dataset & split", "step2_exp": "② Feature selection settings", "step3_exp": "③ Model settings",
        "start_training": "▶ Start Training", "watch_learn": "Watch it learn",

        "rv_eyebrow": "Evaluate & export", "rv_title": "Results & Validation",
        "rv_sub": "How well did it do, and how does it compare to nine other forecasting models?",
        "tab_yourrun": "\U0001F4CA Your run", "tab_bench": "\U0001F52C Published benchmarks",

        "ra_eyebrow": "Research information",
        "ra_sub": "The published paper behind this dashboard, in full — abstract, methodology, selected features, and how this app relates to it.",
        "tab_abstract": "Abstract", "tab_method": "Methodology", "tab_selfeat": "Selected features", "tab_keywords": "Keywords", "tab_about": "About this app",
        "lang_note": "Note: the paper's own abstract, methodology, and data tables below are shown in their original English — this dashboard translates its own interface, not the published paper's technical text.",
    },
    "KO": {
        "brand_sub": "머신러닝 연구실",
        "badge_local": "로컬 · 실시간",
        "nav_Overview": "개요", "nav_Dataset": "데이터셋", "nav_Train & Simulate": "학습",
        "nav_Results & Validation": "결과", "nav_Research & About": "연구",

        "ov_eyebrow": "개요 및 시연",
        "ov_title": "ProphetBoost란 무엇인가요?",
        "ov_p1": "Prophet으로 전력 수요를 추세와 계절성으로 분해한 뒤, XGBoost가 날씨·달력 효과·최근 수요 변화 같은 비선형 잔차를 학습하는 하이브리드 예측 파이프라인입니다. 사용되는 특성은 한 번의 실행에서만 정확한 것이 아니라, 안정적으로 검증된 것만 선별합니다.",
        "ov_p2": "단순히 데이터에 모델 하나를 맞추는 것이 아닙니다. 특성 선택은 부트스트랩으로 검증되므로, 최종적으로 남은 14개의 예측 변수가 우연한 train/test 분할의 결과가 아님을 신뢰할 수 있습니다. 또한 Prophet의 추세·계절성이 명시적이므로, 모든 예측은 '이만큼은 계절 요인, 이만큼은 오늘의 날씨와 최근 수요'처럼 평이한 말로 설명할 수 있습니다.",
        "m_hourly": "시간별 기록 수", "m_feats": "후보 → 선택된 특성", "m_pubmae": "논문 MAE (특성 선택 적용)", "m_datasets": "추적 중인 데이터셋",
        "how_title": "어떻게 동작하나요?",
        "step1_t": "분해", "step1_d": "Prophet이 시간별 수요에서 추세와 연간·주간·일간 계절성을 추출하며, 학습 데이터에만 적합시킵니다.",
        "step2_t": "특성 생성 및 선택", "step2_d": "달력, 지연값, 이동통계와 4개 관측소의 날씨(+지연값)로 111개의 특성을 만들고, XGBoost를 30회 부트스트랩 실행하여 안정적인 14개만 남깁니다.",
        "step3_t": "학습 및 평가", "step3_d": "최종 XGBoost는 안정적인 특성 부분집합으로 교차검증되고, 한 번도 보지 못한 2019-2020년 시간대에서 평가됩니다.",
        "arch_title": "직접 아키텍처를 확인해 보세요",
        "arch_caption": "이것은 목업이 아닙니다 — 실제 논문에 실린 파이프라인 다이어그램이며, 아래의 모든 단계는 이 앱에서 실시간으로 재학습할 수 있습니다.",
        "stat_pubmae": "논문 MAE", "stat_selfeat": "선택된 특성", "stat_improve": "MAE 개선율", "stat_window": "테스트 기간",
        "arch1_t": "1 · Prophet 분해", "arch1_d": "추세 g(t), 푸리에 계절성 s(t), 휴일 효과 h(t), 잔차 노이즈 ε<sub>t</sub>. ĝ(t)와 ŝ(t)만 특성으로 사용되며, 수요 데이터만으로 단변량 적합됩니다.",
        "arch2_t": "2 · 임베디드 + 안정성 선택", "arch2_d": "분산 필터를 거친 뒤 B=30회의 부트스트랩 재표본추출을 수행하며, 매 회 gain 기준 상위 K=15개 특성을 남깁니다. 30회 중 τ·B=18회 이상 선택된 특성만 최종적으로 유지됩니다.",
        "arch3_t": "3 · XGBoost 앙상블", "arch3_d": "이전 잔차를 보정하는 단계적 회귀 트리 앙상블이며, 조기 종료를 적용한 5-겹 시계열 교차검증으로 튜닝됩니다.",
        "arch_footnote": "전체 수식, 선택 알고리즘, 실제로 선택된 14개 특성은 **연구 정보 → Methodology**에서 확인할 수 있습니다.",
        "bench_title": "논문 벤치마크 비교",
        "bench_caption": "ProphetBoost와 9개 기준 모델 비교, 특성 선택 적용 — 논문 Table 9. 전체 표(적용/미적용, 복잡도, 유의성)는 **결과 및 검증**에서 확인할 수 있습니다.",
        "glance_title": "한눈에 보기",
        "feat_title": "이 대시보드로 무엇을 할 수 있나요?",
        "feat1_t": "데이터셋 살펴보기", "feat1_d": "9년간의 제주도 시간별 전력 수요와 날씨 데이터를, 논문에 실린 통계와 함께 살펴봅니다.",
        "feat2_t": "실제 파이프라인 학습하기", "feat2_d": "미리 계산된 데모가 아니라, 원하는 하이퍼파라미터로 실제 Prophet + XGBoost 파이프라인을 직접 실행합니다.",
        "feat3_t": "결과 및 기준 모델 확인하기", "feat3_d": "직접 실행한 결과의 지표와 차트, 그리고 논문에 실린 9개 예측 모델과의 벤치마크 비교를 확인합니다.",
        "feat4_t": "연구 내용 읽어보기", "feat4_d": "수식이 포함된 전체 방법론, 논문에 실린 그림들, 그리고 논문이 실제로 선택한 특성들을 확인합니다.",
        "goto": "바로가기 →",
        "glossary_title": "\U0001F4D6 용어 알아보기 — 이 대시보드에서 만나게 될 6가지 용어",
        "cta_title": "이제 당신의 차례입니다", "cta_d": "기준 날짜를 선택하고 특성 선택 하이퍼파라미터를 조정하면, 동일한 파이프라인이 약 1분 만에 실제 데이터셋으로 학습됩니다.",
        "cta_btn": "직접 학습해보기 →",

        "ds_eyebrow": "데이터셋", "ds_title": "제주도 전력 수요",
        "ds_sub": "디스크의 totalload_new.csv에서 실시간으로 불러옵니다 — 아래 모든 통계는 캐시된 사본이 아닌 실제 파일에서 직접 계산됩니다.",
        "tab_load": "⚡ 전력 수요", "tab_uploaded": "\U0001F4C1 업로드된 데이터셋", "tab_upload_new": "⬆️ 새로 업로드",
        "m_rows": "행", "m_cols": "열", "m_daterange": "기간", "m_loadrange": "수요 범위 (MW)",
        "monthly_load": "월별 평균 수요 (MW)",

        "tr_eyebrow": "모델 학습", "tr_title": "학습 및 시뮬레이션",
        "tr_sub": "실제 데이터셋으로 실제 파이프라인을 학습시키고, 한 번도 본 적 없는 데이터에서 예측이 한 시간씩 펼쳐지는 과정을 지켜보세요.",
        "tab_pb": "⚡ ProphetBoost (실제 파이프라인)", "tab_generic": "\U0001F527 범용 모델 (모든 데이터셋)",
        "step1_exp": "① 데이터셋 및 분할", "step2_exp": "② 특성 선택 설정", "step3_exp": "③ 모델 설정",
        "start_training": "▶ 학습 시작", "watch_learn": "학습 과정 지켜보기",

        "rv_eyebrow": "평가 및 검증", "rv_title": "결과 및 검증",
        "rv_sub": "이 모델은 얼마나 잘 작동했으며, 다른 9개의 예측 모델과 비교하면 어떨까요?",
        "tab_yourrun": "\U0001F4CA 내 실행 결과", "tab_bench": "\U0001F52C 논문 벤치마크",

        "ra_eyebrow": "연구 정보",
        "ra_sub": "이 대시보드의 기반이 된 논문 전체 — 초록, 방법론, 선정된 특성, 그리고 이 앱이 논문과 어떻게 연결되는지 확인하세요.",
        "tab_abstract": "초록", "tab_method": "방법론", "tab_selfeat": "선정된 특성", "tab_keywords": "키워드", "tab_about": "이 앱에 대하여",
        "lang_note": "참고: 아래의 논문 초록·방법론·데이터 표는 원문 그대로 영어로 표시됩니다 — 이 대시보드는 자체 인터페이스만 번역하며, 발표된 논문의 기술적 원문은 번역하지 않습니다.",
    },
}


def t(key):
    return I18N.get(st.session_state.get("lang", "EN"), I18N["EN"]).get(key, I18N["EN"].get(key, key))

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="ProphetBoost Studio", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

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
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


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
    reg = {"Jeju Electricity Load (hourly, 2012-2020)": {"kind": "load"}}
    for rec in read_manifest():
        reg[f"\U0001F4C1 {rec['name']}"] = {"kind": "uploaded", "record": rec}
    return reg


def get_df_for(entry):
    if entry["kind"] == "load":
        return load_load_df(), "datetime"
    df = load_uploaded_df(entry["record"]["stored_filename"])
    date_col = None
    for c in df.columns:
        lc = c.lower()
        if pd.api.types.is_object_dtype(df[c]) and any(k in lc for k in ("date", "time", "_dt")):
            date_col = c
            break
    return df, date_col


def delete_uploaded(rec_id):
    records = read_manifest()
    keep = [r for r in records if r["id"] != rec_id]
    removed = [r for r in records if r["id"] == rec_id]
    write_manifest(keep)
    for r in removed:
        p = UPLOAD_DIR / r["stored_filename"]
        if p.exists():
            p.unlink()
    load_uploaded_df.clear()


# --------------------------------------------------------------------------
# ProphetBoost pipeline (faithful port of mian.ipynb)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_engineered_load_df():
    """Prophet decomposition + feature engineering — identical to mian.ipynb
    cells 1-3. Cached: this is the slow, hyperparameter-independent step."""
    df = load_load_df().copy()
    df = df.rename(columns={"datetime": "ds"})
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
    base_feats = ["trend", "seasonality", "hour", "day_of_week", "is_weekend", "lag_1", "lag_24", "roll3_mean", "roll3_std"]
    lagged_weather = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_lag24")]
    full_feats = base_feats + weather_feats + lagged_weather
    return df, full_feats


def run_prophetboost(cutoff_str, B, K, tau, max_boost_round, early_stopping, progress_cb=None):
    engineered_df, full_feats = build_engineered_load_df()
    cutoff = pd.to_datetime(cutoff_str)
    train = engineered_df[engineered_df["ds"] < cutoff].copy()
    test = engineered_df[engineered_df["ds"] >= cutoff].copy()
    if len(train) < 200 or len(test) < 20:
        raise ValueError("That cutoff leaves too little data on one side of the split.")

    vt = VarianceThreshold(threshold=1e-8)
    vt.fit(train[full_feats])
    filtered_feats = [f for f, keep in zip(full_feats, vt.get_support()) if keep]

    counts = Counter()
    for b in range(B):
        sample = train.sample(frac=1.0, replace=True, random_state=b)
        dtrain_b = xgb.DMatrix(sample[filtered_feats], label=sample["y"])
        model_b = xgb.train(
            {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": 4, "learning_rate": 0.1, "seed": b},
            dtrain_b, num_boost_round=50, verbose_eval=False,
        )
        imp = model_b.get_score(importance_type="gain")
        topk = [f for f, _ in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:K]]
        counts.update(topk)
        if progress_cb:
            progress_cb(b + 1, B)

    selected_feats = [f for f, c in counts.items() if c >= tau * B]
    if not selected_feats:
        selected_feats = [f for f, _ in counts.most_common(5)] or filtered_feats[:5]

    dtrain = xgb.DMatrix(train[selected_feats], label=train["y"])
    dtest = xgb.DMatrix(test[selected_feats], label=test["y"])
    params = {"objective": "reg:squarederror", "tree_method": "hist", "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "mae", "seed": 42}
    cv = xgb.cv(params, dtrain, num_boost_round=max_boost_round, nfold=5, early_stopping_rounds=early_stopping, metrics="mae", as_pandas=True, seed=42)
    best_n = len(cv)
    bst = xgb.train(params, dtrain, num_boost_round=best_n)

    y_pred = bst.predict(dtest)
    y_true = test["y"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    nz = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)

    gain = bst.get_score(importance_type="gain")
    importances = {f: gain.get(f, 0.0) for f in selected_feats}

    df_res = test[["ds", "y"]].copy()
    df_res["pred"] = y_pred

    return {
        "n_candidate_feats": len(full_feats),
        "n_filtered_feats": len(filtered_feats),
        "n_selected_feats": len(selected_feats),
        "selected_feats": selected_feats,
        "best_rounds": best_n,
        "n_train": len(train),
        "n_test": len(test),
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape},
        "importances": importances,
        "df_res": df_res,
    }


# --------------------------------------------------------------------------
# Theming — modeled on PV-Seg Studio (Sora + IBM Plex, indigo accent, pill nav)
# --------------------------------------------------------------------------
def inject_css(dark: bool):
    if dark:
        bg, surface, surface2 = "#0c0e1a", "#12162a", "#181d35"
        text, text2, muted = "#eef0f8", "#b7bdd0", "#7d84a0"
        border = "rgba(255,255,255,0.10)"
        accent, accent_soft, accent_ink = "#7c8cf0", "#232a5c", "#c3caf7"
        good_bg, good_ink = "#123a22", "#4ade80"
        warn_bg, warn_ink = "#3a2c0f", "#f4b73f"
    else:
        bg, surface, surface2 = "#f4f6fb", "#ffffff", "#eef1f8"
        text, text2, muted = "#12172e", "#5b6478", "#8890a0"
        border = "#e3e7f0"
        accent, accent_soft, accent_ink = "#3b4fd6", "#e4e7fc", "#2a3aa8"
        good_bg, good_ink = "#e5f6ea", "#1f8a4c"
        warn_bg, warn_ink = "#fdf3df", "#a86a0a"

    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif; }}
    h1,h2,h3,h4 {{ font-family:'Sora', sans-serif !important; font-weight:700 !important; letter-spacing:-.01em; }}
    code, .mono, [data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono', monospace !important; }}

    #MainMenu {{visibility:hidden;}}
    footer {{visibility:hidden;}}
    [data-testid="collapsedControl"] {{display:none;}}
    section[data-testid="stSidebar"] {{display:none;}}

    [data-testid="stAppViewContainer"] {{ background:{bg}; }}
    [data-testid="stHeader"] {{ display:none; }}
    .stMarkdown, p, li, label, span {{ color:{text}; }}
    h1,h2,h3,h4 {{ color:{text}; }}
    .block-container {{ padding-top:1.2rem; max-width:1180px; }}

    /* ---- top nav ---- */
    .brand {{ display:flex; align-items:center; gap:10px; }}
    .brand .logo {{ width:38px;height:38px;border-radius:11px; background:linear-gradient(155deg,{accent},{accent_ink}); display:flex;align-items:center;justify-content:center; font-size:18px; flex:none; }}
    .brand .t1 {{ font-family:'Sora',sans-serif; font-weight:700; font-size:15.5px; line-height:1.15; color:{text}; }}
    .brand .t2 {{ font-size:9.5px; letter-spacing:.09em; text-transform:uppercase; color:{muted}; }}
    .badge {{ display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:700; padding:6px 12px; border-radius:100px; white-space:nowrap; }}
    .badge-green {{ background:{good_bg}; color:{good_ink}; }}
    .badge-dot {{ width:6px;height:6px;border-radius:50%; background:currentColor; display:inline-block; }}
    div[data-testid="stButton"] button {{ border-radius:100px !important; font-weight:600 !important; font-size:13.5px !important; padding:0.3rem 0.85rem !important; min-width:0 !important; }}
    div[data-testid="stButton"] button[kind="secondary"] {{ border-color:{border} !important; color:{text2} !important; background:{surface} !important; }}
    div[data-testid="stButton"] button[kind="primary"] {{ background:{accent_soft} !important; color:{accent_ink} !important; border-color:{accent_soft} !important; box-shadow:none !important; }}
    hr {{ border-color:{border}; margin:8px 0 22px; }}

    /* ---- page header ---- */
    .eyebrow {{ font-size:11px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:{accent}; margin-bottom:6px; }}
    .pagetitle {{ font-family:'Sora',sans-serif; font-weight:800; font-size:clamp(24px,3vw,32px); color:{text}; letter-spacing:-.01em; }}
    .pagesub {{ color:{text2}; font-size:14.5px; margin-top:6px; max-width:74ch; }}

    /* ---- hero ---- */
    .hero {{ background:linear-gradient(135deg,{surface} 0%,{surface2} 100%); border:1px solid {border}; border-radius:22px; padding:38px 42px; margin-bottom:26px; }}

    /* ---- cards ---- */
    .card {{ background:{surface}; border:1px solid {border}; border-radius:16px; padding:22px 24px; margin-bottom:16px; box-shadow:0 1px 2px rgba(15,23,42,0.03); }}
    .pill {{ display:inline-flex; align-items:center; font-size:11px; font-weight:700; letter-spacing:.02em; padding:4px 11px; border-radius:100px; color:#fff; margin-right:6px; }}
    .pill-neutral {{ background:{surface2}; color:{text2}; }}
    .pill-accent {{ background:{accent_soft}; color:{accent_ink}; }}
    .kw {{ font-size:11px; color:{muted}; background:{surface2}; padding:3px 9px; border-radius:6px; margin:2px 4px 2px 0; display:inline-block; }}
    .muted {{ color:{muted}; font-size:12.5px; }}
    hr {{ border-color:{border}; }}

    /* ---- step / icon list ---- */
    .step {{ display:flex; gap:14px; align-items:flex-start; padding:16px 18px; background:{surface}; border:1px solid {border}; border-radius:14px; }}
    .step .circ {{ width:30px;height:30px;border-radius:50%; background:{accent_soft}; color:{accent_ink}; display:flex; align-items:center; justify-content:center; font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:13px; flex:none; }}
    .step .t {{ font-size:14px; font-weight:700; color:{text}; }}
    .step .d {{ font-size:12.5px; color:{muted}; margin-top:2px; }}

    /* ---- stat tile ---- */
    .stattile {{ background:{surface}; border:1px solid {border}; border-radius:14px; padding:16px 18px; }}
    .stattile .lbl {{ font-size:10.5px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:{muted}; }}
    .stattile .val {{ font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; color:{text}; margin-top:4px; }}
    [data-testid="stMetric"] {{ background:{surface}; border:1px solid {border}; border-radius:14px; padding:14px 18px; box-shadow:0 1px 2px rgba(15,23,42,0.03); }}
    [data-testid="stMetricLabel"] p {{ color:{muted} !important; font-size:10.5px !important; font-weight:700 !important; letter-spacing:.08em !important; text-transform:uppercase !important; }}
    [data-testid="stMetricValue"] {{ color:{text} !important; font-size:22px !important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

    /* ---- feature grid ---- */
    .fcard {{ background:{surface}; border:1px solid {border}; border-radius:16px; padding:20px 22px; height:100%; }}
    .fcard .icon {{ width:34px;height:34px;border-radius:10px; background:{accent_soft}; color:{accent_ink}; display:flex; align-items:center; justify-content:center; font-size:16px; margin-bottom:10px; }}
    .fcard .t {{ font-weight:700; font-size:14.5px; color:{text}; }}
    .fcard .d {{ font-size:12.5px; color:{muted}; margin-top:5px; line-height:1.5; }}

    /* ---- banners ---- */
    .banner {{ border-radius:14px; padding:14px 18px; font-size:13px; display:flex; gap:10px; align-items:flex-start; }}
    .banner-good {{ background:{good_bg}; color:{good_ink}; }}
    .banner-warn {{ background:{warn_bg}; color:{warn_ink}; }}
    .banner b {{ color:inherit; }}

    /* ---- cta ---- */
    .cta {{ background:{surface}; border:1px solid {border}; border-radius:20px; padding:40px; text-align:center; }}

    /* ---- footer ---- */
    .footerbar {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; padding:22px 0 8px; margin-top:30px; border-top:1px solid {border}; font-size:11.5px; color:{muted}; }}

    /* ---- misc streamlit component restyle ---- */
    .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
    .stTabs [aria-selected="true"] {{ color:{accent} !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color:{accent} !important; }}
    [data-testid="stFileUploaderDropzone"] {{ background:{surface2}; border:1.5px dashed {border}; border-radius:14px; }}
    [data-testid="stExpander"] {{ background:{surface}; border:1px solid {border}; border-radius:14px; }}
    .stDownloadButton>button {{ border-radius:9px; font-weight:700; border:1px solid {border}; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


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
# Top navigation (PV-Seg-Studio style: brand + pill tabs + status badges)
# --------------------------------------------------------------------------
NAV_ITEMS = ["Overview", "Dataset", "Train & Simulate", "Results & Validation", "Research & About"]

if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

inject_css(st.session_state.dark_mode)
PT = plotly_theme(st.session_state.dark_mode)

NAV_COL_RATIOS = {
    "EN": [1.9, 1.0, 0.85, 0.75, 0.85, 0.9, 0.78, 0.68, 0.68, 0.45],
    "KO": [1.9, 0.8, 0.95, 0.75, 0.75, 0.75, 0.95, 0.68, 0.68, 0.45],
}
top = st.columns(NAV_COL_RATIOS.get(st.session_state.lang, NAV_COL_RATIOS["EN"]))
with top[0]:
    st.markdown(f'<div class="brand"><div class="logo">⚡</div><div><div class="t1">ProphetBoost</div><div class="t2">{t("brand_sub")}</div></div></div>', unsafe_allow_html=True)
for i, item in enumerate(NAV_ITEMS):
    with top[i + 1]:
        if st.button(t(f"nav_{item}"), key=f"nav_{item}", type="primary" if st.session_state.page == item else "secondary", use_container_width=True):
            st.session_state.page = item
            st.rerun()
with top[6]:
    st.markdown(f'<div style="padding-top:6px;"><span class="badge badge-green"><span class="badge-dot"></span>{t("badge_local")}</span></div>', unsafe_allow_html=True)
with top[7]:
    if st.button("EN", key="lang_en", type="primary" if st.session_state.lang == "EN" else "secondary", use_container_width=True):
        st.session_state.lang = "EN"
        st.rerun()
with top[8]:
    if st.button("KO", key="lang_ko", type="primary" if st.session_state.lang == "KO" else "secondary", use_container_width=True):
        st.session_state.lang = "KO"
        st.rerun()
with top[9]:
    st.toggle("🌙", key="dark_mode", label_visibility="collapsed")
st.markdown("<hr>", unsafe_allow_html=True)


def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="pagetitle">{title}</div><div class="pagesub">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def goto(page_name):
    st.session_state.page = page_name
    st.rerun()


def render_footer():
    st.markdown(
        f"""<div class="footerbar">
        <span>{PAPER['title']}</span>
        <span>Nazrul Amin · Jeju National University</span>
        </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Paper figure rendering (PDF vector figures -> PNG bytes, cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def render_pdf_figure(path_str: str, dpi: int = 200):
    doc = pymupdf.open(path_str)
    pix = doc[0].get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def show_paper_figure(path: Path, caption: str, dpi: int = 200):
    if not path.exists():
        return
    if path.suffix.lower() == ".pdf":
        st.image(render_pdf_figure(str(path), dpi), caption=caption, use_container_width=True)
    else:
        st.image(str(path), caption=caption, use_container_width=True)


# --------------------------------------------------------------------------
# Shared chart builders
# --------------------------------------------------------------------------
def hex_to_rgba(hex_color, alpha=0.12):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def line_chart(x, y, name, color, yaxis_title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2), fill="tozeroy", fillcolor=hex_to_rgba(color)))
    fig.update_layout(**PT, height=320, yaxis_title=yaxis_title, showlegend=False)
    return fig


def bar_chart_h(labels, values, colors, title=""):
    order = np.argsort(values)
    fig = go.Figure(go.Bar(
        x=[values[i] for i in order], y=[labels[i] for i in order], orientation="h",
        marker_color=[colors[i % len(colors)] for i in order],
        text=[f"{values[i]:,.2f}" if abs(values[i]) < 100 else f"{values[i]:,.0f}" for i in order],
        textposition="outside",
    ))
    fig.update_layout(**PT, height=max(220, 34 * len(labels) + 40), title=title, showlegend=False)
    return fig


def grouped_bar(models, series_dict, colors, yaxis_title=""):
    fig = go.Figure()
    for i, (name, vals) in enumerate(series_dict.items()):
        fig.add_trace(go.Bar(x=models, y=vals, name=name, marker_color=colors[i % len(colors)]))
    fig.update_layout(**PT, height=360, barmode="group", yaxis_title=yaxis_title, legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"))
    return fig


def stat_tile(col, label, value):
    col.markdown(f'<div class="stattile"><div class="lbl">{label}</div><div class="val">{value}</div></div>', unsafe_allow_html=True)


# ==========================================================================
# PAGE: Overview
# ==========================================================================
def page_overview():
    st.markdown(
        f"""<div class="hero">
        <div class="eyebrow">{t("ov_eyebrow")}</div>
        <div class="pagetitle" style="font-size:clamp(26px,3.4vw,36px);">{t("ov_title")}</div>
        <p class="pagesub" style="font-size:15.5px;max-width:76ch;">{t("ov_p1")}</p>
        <p class="pagesub">{t("ov_p2")}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    n_uploaded = len(read_manifest())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("m_hourly"), "75,983", help="2012-01-01 → 2020-12-08")
    c2.metric(t("m_feats"), f"{N_CANDIDATE_FEATS_REF} → {N_SELECTED_FEATS_REF}")
    c3.metric(t("m_pubmae"), f"{MAE_WITH_FS_REF:.2f} MW", help=f"{MAE_IMPROVEMENT_PCT}% — {MAE_WITHOUT_FS_REF} MW without FS")
    c4.metric(t("m_datasets"), str(1 + n_uploaded))

    st.write("")
    st.markdown(f"#### {t('how_title')}")
    steps = [
        ("1", t("step1_t"), t("step1_d")),
        ("2", t("step2_t"), t("step2_d")),
        ("3", t("step3_t"), t("step3_d")),
    ]
    cols = st.columns(3)
    for col, (n, st_title, st_desc) in zip(cols, steps):
        col.markdown(f'<div class="step"><div class="circ">{n}</div><div><div class="t">{st_title}</div><div class="d">{st_desc}</div></div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown(f"#### {t('arch_title')}")
    st.caption(t("arch_caption"))
    show_paper_figure(FIG_PIPELINE, "Fig. 2 (paper) — full ProphetBoost pipeline: preprocessing → Prophet decomposition → feature engineering → embedded+stability selection → XGBoost ensemble → evaluation & SHAP.")

    s1, s2, s3, s4 = st.columns(4)
    stat_tile(s1, t("stat_pubmae"), f"{MAE_WITH_FS_REF} MW")
    stat_tile(s2, t("stat_selfeat"), f"{N_SELECTED_FEATS_REF} / {N_CANDIDATE_FEATS_REF}")
    stat_tile(s3, t("stat_improve"), f"{MAE_IMPROVEMENT_PCT}%")
    stat_tile(s4, t("stat_window"), "2019–2020")

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            f"""<div class="card" style="height:100%;">
            <b>{t("arch1_t")}</b>
            <p class="muted" style="margin-top:8px;">y(t) = g(t) + s(t) + h(t) + ε<sub>t</sub></p>
            <p class="muted">{t("arch1_d")}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"""<div class="card" style="height:100%;">
            <b>{t("arch2_t")}</b>
            <p class="muted" style="margin-top:8px;">{t("arch2_d")}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            f"""<div class="card" style="height:100%;">
            <b>{t("arch3_t")}</b>
            <p class="muted" style="margin-top:8px;">ŷ<sub>i</sub><sup>(t)</sup> = Σ<sub>k=1..t</sub> f<sub>k</sub>(x<sub>i</sub>)</p>
            <p class="muted">{t("arch3_d")}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    st.caption(t("arch_footnote"))

    st.write("")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"#### {t('bench_title')}")
        st.caption(t("bench_caption"))
        models = list(ACCURACY_WITH_FS.keys())
        mae_vals = [ACCURACY_WITH_FS[m]["MAE"] for m in models]
        st.plotly_chart(bar_chart_h(models, mae_vals, CATEGORICAL, "Test MAE (MW)"), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.markdown(f"#### {t('glance_title')}")
        st.markdown(
            f"""<div class="card">
            <b>⚡ Jeju Electricity Load dataset</b><br>
            <span class="muted">75,983 hourly rows · 2012–2020 · 4 weather stations (Jeju, Gosan, Sungsan, Seogwipo)</span>
            </div>
            <div class="card">
            <b>\U0001F4C4 {PAPER['title'][:40]}…</b><br>
            <span class="muted">{PAPER['venue']} · {PAPER['citation']}</span><br>
            <span class="muted">{PAPER['authors']}</span><br>
            <a href="https://doi.org/{PAPER['doi']}" target="_blank" style="font-size:12px;">doi.org/{PAPER['doi']}</a>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(f"#### {t('feat_title')}")
    feats = [
        ("\U0001F5C2️", t("feat1_t"), t("feat1_d"), "Dataset"),
        ("\U0001F9EA", t("feat2_t"), t("feat2_d"), "Train & Simulate"),
        ("\U0001F4C8", t("feat3_t"), t("feat3_d"), "Results & Validation"),
        ("\U0001F4C4", t("feat4_t"), t("feat4_d"), "Research & About"),
    ]
    g1, g2 = st.columns(2)
    for i, (icon, title, desc, target) in enumerate(feats):
        col = g1 if i % 2 == 0 else g2
        with col:
            st.markdown(f'<div class="fcard"><div class="icon">{icon}</div><div class="t">{title}</div><div class="d">{desc}</div></div>', unsafe_allow_html=True)
            if st.button(t("goto"), key=f"goto_{target}"):
                goto(target)

    st.write("")
    with st.expander(t("glossary_title")):
        for term, desc in GLOSSARY.get(st.session_state.lang, GLOSSARY["EN"]):
            st.markdown(f"**{term}** — {desc}")

    st.write("")
    st.markdown(
        f"""<div class="cta">
        <h3 style="margin-bottom:6px;">{t("cta_title")}</h3>
        <p class="muted" style="font-size:14px;">{t("cta_d")}</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    cta1, cta2, cta3 = st.columns([1, 1, 1])
    with cta2:
        if st.button(t("cta_btn"), key="cta_train", type="primary", use_container_width=True):
            goto("Train & Simulate")


# ==========================================================================
# PAGE: Dataset
# ==========================================================================
def page_dataset():
    page_header(t("ds_eyebrow"), t("ds_title"), t("ds_sub"))

    load_tab, uploads_tab, upload_new_tab = st.tabs([t("tab_load"), t("tab_uploaded"), t("tab_upload_new")])

    with load_tab:
        df = load_load_df()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("m_rows"), f"{len(df):,}")
        c2.metric(t("m_cols"), len(df.columns))
        c3.metric(t("m_daterange"), f"{df['datetime'].min():%Y} → {df['datetime'].max():%Y}", help=f"{df['datetime'].min():%Y-%m-%d} → {df['datetime'].max():%Y-%m-%d}")
        c4.metric(t("m_loadrange"), f"{df['TOTAL_LOAD'].min():.0f}–{df['TOTAL_LOAD'].max():.0f}")

        st.write("")
        monthly = df.set_index("datetime")["TOTAL_LOAD"].resample("MS").mean()
        st.markdown(f"**{t('monthly_load')}**")
        st.plotly_chart(line_chart(monthly.index, monthly.values, "Avg. load", SERIES_BLUE, "MW"), use_container_width=True, config={"displayModeBar": False})

        with st.expander("Column reference"):
            weather_cols = [c for c in df.columns if any(c.startswith(p) for p in WEATHER_PREFIXES)]
            st.write(f"**Weather (×4 stations, {len(weather_cols)} columns):**", ", ".join(weather_cols))
            st.write("**Target:** TOTAL_LOAD (MW, hourly)")

        with st.expander("Raw data preview"):
            st.dataframe(df.head(50), use_container_width=True)
        st.download_button("Download full CSV", LOAD_CSV.read_bytes(), file_name=LOAD_CSV.name, mime="text/csv")

        st.write("")
        st.markdown("#### Published dataset statistics (paper Tables 3-5)")
        st.caption("As reported in the paper — computed on its own preprocessed 2012-2020 partition, so figures differ slightly from the live stats above.")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**Load statistics**")
            stats_df = pd.DataFrame(
                [("Mean", f"{DATASET_STATS_REF['mean']} MW"), ("Std. deviation", f"{DATASET_STATS_REF['std']} MW"),
                 ("Min", f"{DATASET_STATS_REF['min']:.0f} MW"), ("Max", f"{DATASET_STATS_REF['max']:.0f} MW"),
                 ("Coefficient of variation", f"{DATASET_STATS_REF['cv']}%")],
                columns=["Metric", "Value"],
            )
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.markdown("**Train / test partition**")
            part_df = pd.DataFrame(PARTITION_REF, columns=["Split", "Period", "Samples"])
            st.dataframe(part_df, use_container_width=True, hide_index=True)
        with pcol2:
            st.markdown("**Strongest correlations with load**")
            corr_rows = [(f, c, "positive") for f, c in TOP_CORRELATIONS_REF["positive"]] + [(f, c, "negative") for f, c in TOP_CORRELATIONS_REF["negative"]]
            corr_df = pd.DataFrame(corr_rows, columns=["Feature", "Correlation", "Direction"])
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
            st.caption("All |r| ≤ 0.18 — weather's relationship to load is real but weak and nonlinear, which is why the paper motivates a tree-based model (XGBoost) rather than a linear one.")

    with uploads_tab:
        records = read_manifest()
        if not records:
            st.markdown('<div class="banner banner-warn">No datasets uploaded yet. Use the <b>Upload new</b> tab to add one.</div>', unsafe_allow_html=True)
        for rec in records:
            with st.container(border=True):
                top_r = st.columns([4, 1])
                top_r[0].markdown(f"**{rec['name']}**")
                top_r[0].caption(f"{rec['original_filename']} · added {rec['uploaded_at'][:10]}")
                if top_r[1].button("Remove", key=f"del_{rec['id']}"):
                    delete_uploaded(rec["id"])
                    st.rerun()
                try:
                    df = load_uploaded_df(rec["stored_filename"])
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Rows", f"{len(df):,}")
                    cc2.metric("Columns", len(df.columns))
                    if rec.get("note"):
                        st.caption(rec["note"])
                    st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read this file: {e}")

    with upload_new_tab:
        st.caption("Adds a CSV to the local catalog (saved under `uploaded_datasets/`). Browse it above, or train the generic model on it from **Train & Simulate → Generic model**.")
        file = st.file_uploader("Drop a CSV file", type=["csv"])
        if file is not None:
            try:
                df = pd.read_csv(file)
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin-1")

            st.markdown(f'<div class="banner banner-good">✓ Parsed {len(df):,} rows × {len(df.columns)} columns.</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", len(df.columns))
            c3.metric("Missing (avg.)", f"{df.isna().mean().mean()*100:.1f}%")

            with st.expander("Column summary", expanded=True):
                summary = pd.DataFrame({"type": df.dtypes.astype(str), "missing %": (df.isna().mean() * 100).round(1), "unique": df.nunique()})
                st.dataframe(summary, use_container_width=True)

            st.markdown("**Preview**")
            st.dataframe(df.head(10), use_container_width=True)

            name = st.text_input("Dataset name", value=Path(file.name).stem)
            note = st.text_input("Notes (optional)", placeholder="Source, units, related paper…")

            if st.button("Add to catalog", type="primary"):
                stored_filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}_{file.name}"
                file.seek(0)
                (UPLOAD_DIR / stored_filename).write_bytes(file.read())
                records = read_manifest()
                records.append({
                    "id": uuid.uuid4().hex, "name": name or file.name, "note": note,
                    "original_filename": file.name, "stored_filename": stored_filename,
                    "uploaded_at": datetime.now().isoformat(), "rows": len(df), "columns": len(df.columns),
                })
                write_manifest(records)
                st.success(f"'{name}' added.")
                st.balloons()


# ==========================================================================
# Generic trainer (for uploaded datasets)
# ==========================================================================
MODEL_BUILDERS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda n, d: RandomForestRegressor(n_estimators=n, max_depth=d, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda n, d: GradientBoostingRegressor(n_estimators=n, max_depth=min(d, 6), random_state=42),
}


def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def default_feature_cols_generic(df, target):
    return [c for c in numeric_columns(df) if c != target]


# ==========================================================================
# PAGE: Train & Simulate
# ==========================================================================
def page_train_and_simulate():
    page_header(t("tr_eyebrow"), t("tr_title"), t("tr_sub"))
    tab_pb, tab_generic = st.tabs([t("tab_pb"), t("tab_generic")])
    with tab_pb:
        page_train_prophetboost()
    with tab_generic:
        page_train_generic()


def page_train_prophetboost():
    st.caption("Six steps, no code: the exact ProphetBoost pipeline from the project notebook, fit live on totalload_new.csv.")

    with st.expander(t("step1_exp"), expanded=True):
        st.markdown('<div class="banner banner-good">✓ Using the laboratory dataset already on this server — <b>totalload_new.csv</b>, 75,983 hourly rows.</div>', unsafe_allow_html=True)
        cutoff = st.date_input("Train / test cutoff date", value=pd.to_datetime("2019-01-01"), min_value=pd.to_datetime("2012-06-01"), max_value=pd.to_datetime("2020-06-01"))
        st.caption("Rows before this date train the model; rows on/after it are the held-out test set. Paper default: 2019-01-01.")

    with st.expander(t("step2_exp"), expanded=True):
        c3, c4, c5 = st.columns(3)
        B = c3.slider("Bootstrap resamples (B)", 5, 30, 15, help="Paper uses 30. Fewer = faster, less stable selection.")
        K = c4.slider("Top-K per resample", 5, 30, 15)
        tau = c5.slider("Stability threshold (τ)", 0.3, 0.9, 0.6, step=0.05, help="Keep a feature if it's in the top-K in ≥ τ·B resamples.")

    with st.expander(t("step3_exp"), expanded=True):
        c6, c7 = st.columns(2)
        max_rounds = c6.slider("Max CV boosting rounds", 100, 1000, 500, step=100)
        early_stop = c7.slider("Early stopping rounds", 10, 50, 20)

    est_secs = int(B * 1.8 + 20)
    st.markdown(f'<div class="banner banner-warn">⚠ Estimated runtime: ~{est_secs}s. The first run also fits Prophet once (cached afterward) — that alone can take 20-60s.</div>', unsafe_allow_html=True)
    st.write("")

    trained_now = False
    if st.button(t("start_training"), type="primary"):
        prog = st.progress(0.0, text="Preparing engineered features (Prophet decomposition — cached after first run)…")
        try:
            engineered_df, _ = build_engineered_load_df()
        except Exception as e:
            st.error(f"Feature engineering failed: {e}")
            return
        prog.progress(0.15, text=f"Engineered {len(engineered_df):,} rows. Running stability feature selection…")

        def cb(b, total):
            frac = 0.15 + 0.55 * (b / total)
            prog.progress(frac, text=f"Stability selection: bootstrap {b}/{total}")

        t0 = time.time()
        try:
            result = run_prophetboost(str(cutoff), B, K, tau, max_rounds, early_stop, progress_cb=cb)
        except Exception as e:
            prog.empty()
            st.error(f"Training failed: {e}")
            return
        prog.progress(0.9, text="Cross-validating & fitting final XGBoost…")
        elapsed = time.time() - t0
        prog.progress(1.0, text="Done.")
        time.sleep(0.3)
        prog.empty()

        run = {
            "id": uuid.uuid4().hex[:8],
            "label": f"ProphetBoost · cutoff {cutoff} · B={B}",
            "kind": "prophetboost",
            "elapsed": elapsed,
            "n_train": result["n_train"], "n_test": result["n_test"],
            "metrics": result["metrics"],
            "dates_test": [str(d) for d in result["df_res"]["ds"]],
            "y_test": result["df_res"]["y"].tolist(),
            "y_pred": result["df_res"]["pred"].tolist(),
            "importances": result["importances"],
            "n_candidate_feats": result["n_candidate_feats"],
            "n_selected_feats": result["n_selected_feats"],
            "trained_at": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.setdefault("runs", [])
        st.session_state.runs.insert(0, run)
        st.session_state.runs = st.session_state.runs[:8]
        st.session_state["last_pb_run_id"] = run["id"]
        trained_now = True

    runs = st.session_state.get("runs", [])
    pb_runs = [r for r in runs if r.get("kind") == "prophetboost"]

    st.write("")
    st.markdown(f"#### {t('watch_learn')}")
    if not pb_runs:
        st.markdown(
            """<div class="card" style="text-align:center;padding:44px 20px;">
            <div style="font-size:26px;">\U0001F4C8</div>
            <b>No training history yet</b>
            <p class="muted" style="margin-top:6px;">Press Start Training above. Metrics, the actual-vs-predicted chart, and a live interval-by-interval reveal appear here once a run finishes.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    default_run = pb_runs[0]
    m = default_run["metrics"]
    if trained_now:
        st.markdown(f'<div class="banner banner-good">✓ Trained in {default_run["elapsed"]:.1f}s · {default_run["n_selected_feats"]} of {default_run["n_candidate_feats"]} features selected · {default_run["n_train"]:,} train / {default_run["n_test"]:,} test rows.</div>', unsafe_allow_html=True)
        st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{m['MAE']:.2f} MW")
    m2.metric("RMSE", f"{m['RMSE']:.2f} MW")
    m3.metric("R²", f"{m['R2']:.3f}")
    m4.metric("MAPE", f"{m['MAPE']:.2f}%")
    st.caption("Full charts, feature importance, and a day/month error breakdown are on **Results & Validation**.")

    st.write("")
    render_live_simulation(pb_runs, key_prefix="pb_inline")


# --------------------------------------------------------------------------
# Live simulation (embedded inline on Train & Simulate, ProphetBoost tab)
# --------------------------------------------------------------------------
def pick_run(runs, key="pick_run"):
    labels = [f"{r['label']} · trained {r['trained_at']}" for r in runs]
    sel = st.selectbox("Model run", range(len(runs)), format_func=lambda i: labels[i], key=key)
    return runs[sel]


def render_live_simulation(runs, key_prefix="sim"):
    st.markdown("##### Live simulation — actual vs. predicted, revealed interval by interval")
    run = pick_run(runs, key=f"{key_prefix}_pick")
    y_test = np.array(run["y_test"])
    y_pred = np.array(run["y_pred"])

    window = min(150, len(y_test))
    y_test_w = y_test[-window:]
    y_pred_w = y_pred[-window:]
    if window < len(y_test):
        st.caption(f"Showing the most recent {window} of {len(y_test)} test hours for a readable animation.")

    idx_key = f"{key_prefix}_idx_{run['id']}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = max(2, window // 10)

    c1, c2, c3 = st.columns([3, 1, 1])
    st.session_state[idx_key] = c1.slider("Reveal up to interval", 2, window, st.session_state[idx_key], key=f"{key_prefix}_slider_{run['id']}")
    play = c2.button("▶ Play", key=f"{key_prefix}_play_{run['id']}")
    reset = c3.button("↺ Reset", key=f"{key_prefix}_reset_{run['id']}")
    if reset:
        st.session_state[idx_key] = 2
        st.rerun()

    chart_ph = st.empty()
    metric_ph = st.empty()
    unit = "MW" if run.get("kind") == "prophetboost" else run.get("target", "")

    def render(i):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(i)), y=y_test_w[:i], name="Actual", mode="lines", line=dict(color="#8890a0", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=list(range(i)), y=y_pred_w[:i], name="Predicted", mode="lines+markers", line=dict(color=SERIES_BLUE, width=2), marker=dict(size=4)))
        fig.update_layout(**PT, height=340, xaxis_title="Test hour", yaxis_title=unit, showlegend=True, legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
        chart_ph.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        running_mae = float(np.mean(np.abs(y_test_w[:i] - y_pred_w[:i])))
        with metric_ph.container():
            mc1, mc2 = st.columns(2)
            mc1.metric("Intervals revealed", f"{i} / {window}")
            mc2.metric("Running MAE", f"{running_mae:,.2f}")

    if play:
        for i in range(st.session_state[idx_key], window + 1, max(1, window // 60)):
            render(i)
            time.sleep(0.06)
        st.session_state[idx_key] = window
        render(window)
    else:
        render(st.session_state[idx_key])


def page_train_generic():
    st.caption("A lighter scikit-learn regressor for any dataset — pick a target, pick features, pick a model.")
    reg = dataset_registry()
    label = st.selectbox("Dataset", list(reg.keys()), key="generic_ds")
    entry = reg[label]
    df, date_col = get_df_for(entry)

    if len(df) < 30 or len(numeric_columns(df)) < 2:
        st.warning("This dataset needs at least 30 rows and 2 numeric columns to train on.")
        return

    num_cols = numeric_columns(df)
    default_target = "TOTAL_LOAD" if entry["kind"] == "load" else num_cols[-1]
    target = st.selectbox("Target column", num_cols, index=num_cols.index(default_target) if default_target in num_cols else 0, key="generic_target")

    default_feats = default_feature_cols_generic(df, target)
    feature_cols = st.multiselect("Feature columns", [c for c in num_cols if c != target], default=default_feats[:15], key="generic_feats")

    add_calendar = False
    if date_col and date_col in df.columns:
        add_calendar = st.checkbox("Add calendar features (month, day-of-week, hour)", value=True, key="generic_cal")

    col_a, col_b, col_c = st.columns(3)
    model_name = col_a.selectbox("Model", list(MODEL_BUILDERS.keys()), key="generic_model")
    test_pct = col_b.slider("Test set size (most recent %)", 10, 40, 20, key="generic_test_pct")
    max_rows = col_c.number_input("Max training rows (speed cap)", min_value=500, max_value=int(len(df)), value=min(len(df), 20000), step=500, key="generic_max_rows")

    if model_name in ("Random Forest", "Gradient Boosting"):
        col_d, col_e = st.columns(2)
        n_estimators = col_d.slider("Number of trees", 20, 300, 120, step=10, key="generic_n_est")
        max_depth = col_e.slider("Max tree depth", 3, 25, 12, key="generic_max_depth")
    else:
        n_estimators, max_depth = None, None

    if not feature_cols:
        st.warning("Pick at least one feature column.")
        return

    if st.button("▶ Train generic model", type="primary"):
        with st.spinner(f"Training {model_name} on {label}…"):
            work = df.copy()
            if len(work) > max_rows:
                work = work.tail(int(max_rows)).reset_index(drop=True)

            X = work[feature_cols].copy()
            if add_calendar and date_col:
                dt = pd.to_datetime(work[date_col])
                X["__month"] = dt.dt.month
                X["__dow"] = dt.dt.dayofweek
                X["__hour"] = dt.dt.hour
            y = work[target]

            valid = X.notna().all(axis=1) & y.notna()
            dropped = (~valid).sum()
            X, y = X[valid], y[valid]
            if date_col:
                dates = pd.to_datetime(work.loc[valid, date_col]).reset_index(drop=True)
            else:
                dates = pd.Series(range(len(X)))

            if len(X) < 20:
                st.error("Too few complete rows after dropping missing values.")
                return

            split = int(len(X) * (1 - test_pct / 100))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            dates_test = dates.iloc[split:]

            t0 = time.time()
            model = MODEL_BUILDERS[model_name]() if model_name == "Linear Regression" else MODEL_BUILDERS[model_name](n_estimators, max_depth)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elapsed = time.time() - t0

            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            r2 = r2_score(y_test, y_pred)
            nz = y_test != 0
            mape = float(np.mean(np.abs((y_test[nz] - y_pred[nz]) / y_test[nz])) * 100) if nz.any() else float("nan")

            importances = model.feature_importances_ if hasattr(model, "feature_importances_") else (lambda c: c / (c.sum() or 1))(np.abs(model.coef_))

            run = {
                "id": uuid.uuid4().hex[:8],
                "label": f"{model_name} → {target} ({label.split('(')[0].strip()})",
                "kind": "generic",
                "elapsed": elapsed, "n_train": len(X_train), "n_test": len(X_test),
                "dropped_rows": int(dropped),
                "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape},
                "dates_test": [str(d) for d in dates_test],
                "y_test": y_test.tolist(), "y_pred": y_pred.tolist(),
                "importances": dict(zip(X.columns, importances.tolist())),
                "target": target, "trained_at": datetime.now().strftime("%H:%M:%S"),
            }
            st.session_state.setdefault("runs", [])
            st.session_state.runs.insert(0, run)
            st.session_state.runs = st.session_state.runs[:8]

        st.markdown(f'<div class="banner banner-good">✓ Trained in {elapsed:.1f}s on {run["n_train"]:,} rows · tested on {run["n_test"]:,} rows.</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{mae:,.2f}")
        m2.metric("RMSE", f"{rmse:,.2f}")
        m3.metric("R²", f"{r2:.3f}")
        m4.metric("MAPE", f"{mape:.1f}%" if not np.isnan(mape) else "—")

    generic_runs = [r for r in st.session_state.get("runs", []) if r.get("kind") == "generic"]
    if generic_runs:
        st.write("")
        render_live_simulation(generic_runs, key_prefix="generic_inline")


# ==========================================================================
# PAGE: Results & Validation
# ==========================================================================
def page_results_and_validation():
    page_header(t("rv_eyebrow"), t("rv_title"), t("rv_sub"))
    tab_run, tab_bench = st.tabs([t("tab_yourrun"), t("tab_bench")])
    with tab_run:
        page_results()
    with tab_bench:
        page_external_validation()


def page_results():
    runs = st.session_state.get("runs", [])
    if not runs:
        st.markdown(
            """<div class="card" style="text-align:center;padding:44px 20px;">
            <div style="font-size:26px;">\U0001F4CA</div>
            <b>Not trained yet</b>
            <p class="muted" style="margin-top:6px;">Go to <b>Train & Simulate</b> and press Start Training. It runs the pipeline on the real dataset and collects the scores and figures below.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    run = pick_run(runs, key="results_pick")
    m = run["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{m['MAE']:,.2f}")
    c2.metric("RMSE", f"{m['RMSE']:,.2f}")
    c3.metric("R²", f"{m['R2']:.3f}")
    c4.metric("MAPE", f"{m['MAPE']:.2f}%" if not np.isnan(m["MAPE"]) else "—")

    extra = f" · {run['n_selected_feats']} of {run['n_candidate_feats']} features selected" if run.get("kind") == "prophetboost" else ""
    st.caption(f"{run['label']} · {run['n_train']:,} train / {run['n_test']:,} test rows{extra} · trained in {run['elapsed']:.1f}s")

    if run.get("kind") == "prophetboost":
        delta = m["MAE"] - MAE_WITH_FS_REF
        st.caption(f"Published paper (Table 7, with FS): MAE {MAE_WITH_FS_REF} MW · this run: {m['MAE']:.2f} MW ({'+' if delta >= 0 else ''}{delta:.2f} MW) — differences come from the hyperparameters chosen and the exact cutoff date.")

    y_test = np.array(run["y_test"])
    y_pred = np.array(run["y_pred"])
    x = list(range(len(y_test)))

    st.markdown("#### Actual vs. predicted (full test set)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_test, name="Actual", mode="lines", line=dict(color="#8890a0", width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=y_pred, name="Predicted", mode="lines", line=dict(color=SERIES_BLUE, width=2)))
    fig.update_layout(**PT, height=360, xaxis_title="Test interval", yaxis_title="Load (MW)" if run.get("kind") == "prophetboost" else "")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature importance (gain)")
        imp = sorted(run["importances"].items(), key=lambda kv: kv[1], reverse=True)[:12]
        st.plotly_chart(bar_chart_h([k for k, _ in imp], [v for _, v in imp], CATEGORICAL), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.markdown("#### Residuals")
        residuals = y_test - y_pred
        fig2 = go.Figure(go.Histogram(x=residuals, marker_color=SERIES_ORANGE, nbinsx=30))
        fig2.update_layout(**PT, height=340, xaxis_title="Actual − Predicted", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    if run.get("kind") == "prophetboost" and run.get("dates_test"):
        st.markdown("#### Error by day-of-week & month")
        st.caption("Computed live from this run's test predictions — same method as the project notebook.")
        res_df = pd.DataFrame({"ds": pd.to_datetime(run["dates_test"]), "y": y_test, "pred": y_pred})
        res_df["abs_err"] = (res_df["y"] - res_df["pred"]).abs()
        res_df["Day"] = res_df["ds"].dt.day_name()
        res_df["Month"] = res_df["ds"].dt.month_name()

        cA, cB = st.columns(2)
        with cA:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_mae = res_df.groupby("Day")["abs_err"].mean().reindex(day_order)
            st.plotly_chart(bar_chart_h(list(daily_mae.index), list(daily_mae.values), CATEGORICAL, "MAE by day"), use_container_width=True, config={"displayModeBar": False})
        with cB:
            month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            monthly_mae = res_df.groupby("Month")["abs_err"].mean().reindex(month_order).dropna()
            st.plotly_chart(bar_chart_h(list(monthly_mae.index), list(monthly_mae.values), CATEGORICAL, "MAE by month"), use_container_width=True, config={"displayModeBar": False})

    if run.get("kind") == "prophetboost":
        with st.expander("Compare against the published paper's diagnostic panel"):
            show_paper_figure(FIG_MODEL_DIAGNOSTICS, "Fig. 8 (paper) — learning curve, actual vs. predicted, residuals, and top-10 gain importances from the published run.")

    pred_df = pd.DataFrame({"index": x, "actual": y_test, "predicted": y_pred, "residual": y_test - y_pred})
    st.download_button("Download predictions as CSV", pred_df.to_csv(index=False).encode(), file_name=f"predictions_{run['id']}.csv", mime="text/csv")

    st.markdown("#### Run history (this session)")
    hist = pd.DataFrame([{"Run": r["label"], "Trained": r["trained_at"], "MAE": round(r["metrics"]["MAE"], 2), "R²": round(r["metrics"]["R2"], 3), "Train rows": r["n_train"]} for r in runs])
    st.dataframe(hist, use_container_width=True, hide_index=True)


def page_external_validation():
    st.caption(
        "ProphetBoost compared against nine baselines spanning statistical, linear, deep, and attention-based "
        "paradigms (ARIMA-LSTM, DLinear, CNN-ANN, Deep RVFL, CNN-LSTM, Transformer, DeepAR, Autoformer, TFT) — "
        "paper Tables 9, 11 and 12. These need PyTorch + Optuna hyperparameter search and take minutes to hours "
        "to train, so this section shows the published paper's own results rather than retraining them live — "
        "everything under **Your run**, by contrast, runs live in this app."
    )

    fs_on = st.toggle("Feature selection applied", value=True, help="Off shows the same comparison without the stability feature-selection step (paper's 'Without FS' columns).")
    data = ACCURACY_WITH_FS if fs_on else ACCURACY_WITHOUT_FS
    comp_data = COMPLEXITY_WITH_FS if fs_on else COMPLEXITY_WITHOUT_FS

    st.markdown(f'<div class="banner banner-good">Headline result (paper Table 7): feature selection cuts ProphetBoost\'s MAE by <b>{MAE_IMPROVEMENT_PCT}%</b> — from {MAE_WITHOUT_FS_REF} MW to {MAE_WITH_FS_REF} MW.</div>', unsafe_allow_html=True)
    st.write("")

    models = list(data.keys())
    st.markdown("#### Error metrics")
    st.plotly_chart(
        grouped_bar(models, {"MAE": [data[m]["MAE"] for m in models], "RMSE": [data[m]["RMSE"] for m in models]}, [SERIES_BLUE, SERIES_ORANGE], "MW"),
        use_container_width=True, config={"displayModeBar": False},
    )

    table = pd.DataFrame(data).T[["MAE", "RMSE", "MAPE"]].round(3)
    table.columns = ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]
    st.dataframe(table, use_container_width=True)

    st.markdown("#### Model complexity")
    st.caption(f"Training/inference cost on an NVIDIA RTX 4060 (8 GB), {'with' if fs_on else 'without'} feature selection.")
    comp = pd.DataFrame(comp_data).T[["train_s", "infer_s", "infer_ms", "size_mb", "gpu_mb"]].round(4)
    comp.columns = ["Train time (s)", "Inference time (s)", "Infer / sample (ms)", "Model size (MB)", "GPU memory (MB)"]
    st.dataframe(comp, use_container_width=True)

    st.markdown("#### Statistical significance")
    st.caption("Wilcoxon signed-rank test: ProphetBoost's errors vs. each baseline's, on the same 2019-2020 test hours (paper Table 12, with feature selection).")
    wdf = pd.DataFrame(WILCOXON_VS_PROPHETBOOST, columns=["Compared model", "W statistic", "p-value"])
    wdf["Significant (α=0.05)"] = wdf["p-value"] < 0.05
    st.dataframe(wdf, use_container_width=True, hide_index=True)
    st.markdown(f"**Friedman test:** statistic = `{FRIEDMAN['statistic']:.4f}`, p = `{FRIEDMAN['p']:.6f}` → {FRIEDMAN['decision']}.")

    st.markdown("#### Daily / monthly error (published ProphetBoost run, with feature selection)")
    dcol, mcol = st.columns(2)
    with dcol:
        ddf = pd.DataFrame(DAILY_METRICS, columns=["Day", "MAE", "MSE", "RMSE", "MAPE (%)"])
        st.dataframe(ddf, use_container_width=True, hide_index=True)
    with mcol:
        mdf = pd.DataFrame(MONTHLY_METRICS, columns=["Month", "MAE", "MSE", "RMSE", "MAPE (%)"])
        st.dataframe(mdf, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: Research & About
# ==========================================================================
def page_research_and_about():
    page_header(t("ra_eyebrow"), PAPER["title"][:52] + ("…" if len(PAPER["title"]) > 52 else ""), t("ra_sub"))

    st.markdown(
        f"""<div class="card">
        <span class="pill pill-accent">Energy Forecasting</span>
        <span class="pill pill-neutral">{PAPER['venue']}</span>
        <h3 style="margin-top:14px;">{PAPER['title']}</h3>
        <p class="muted"><b style="color:inherit;">{PAPER['authors'].split(',')[0]}</b>{','.join(PAPER['authors'].split(',')[1:])}</p>
        <p class="muted">{PAPER['affiliation']} · Advisor: <b style="color:inherit;">{PAPER['advisor']}</b></p>
        <p class="muted">{PAPER['citation']} · <a href="https://doi.org/{PAPER['doi']}" target="_blank">doi.org/{PAPER['doi']}</a></p>
        <p class="muted">{PAPER['dates']}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([t("tab_abstract"), t("tab_method"), t("tab_selfeat"), t("tab_keywords"), t("tab_about")])
    if st.session_state.lang == "KO":
        st.caption(t("lang_note"))
    with tab1:
        st.write(PAPER["abstract"])
        st.markdown("**Conclusion.** " + PAPER["conclusion"])
    with tab2:
        st.markdown(
            """
**1. Data preparation.** Empty columns dropped, `BASE_DT` parsed to a datetime index, remaining numeric
columns (including the target) mean-imputed. Outliers are IQR-bounded, but genuine load peaks are kept —
they're real peak-demand events, not noise.

**2. Prophet decomposition.**
"""
        )
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.markdown(
            r"""
$g(t)$ is a (piecewise logistic or linear) trend with changepoints, $s(t)$ a Fourier-series seasonal
component (daily / weekly / yearly), $h(t)$ holiday effects, and $\epsilon_t \sim \mathcal{N}(0,\sigma^2)$
residual noise. Prophet is fit **univariately** on the load series alone — weather is not passed in as a
Prophet regressor; it enters later, in the XGBoost stage. Only the fitted trend $\hat g(t)$ and yearly
seasonality $\hat s(t)$ are kept as engineered features.

**3. Feature engineering.** Calendar features (hour, day-of-week, weekend flag), load dynamics (1h and
24h lags, 3h rolling mean/std), and 36 raw weather covariates from four stations (9 variables each:
temperature, dew point, humidity, wind speed/direction, pressure, dew-point index, snow/ground metric,
solar irradiance) — each weather variable also lagged 1h and 24h. Together: **111 candidate features**.
Rows are split chronologically first (train `< 2019-01-01`, test `≥ 2019-01-01`), and Prophet is fit
*only* on the training window to avoid leakage.

**4. Embedded + stability feature selection.** A two-stage XAI-driven filter:
"""
        )
        st.latex(r"\mathrm{Var}(X_j) < 10^{-8} \Rightarrow \text{drop}")
        st.markdown(
            r"""
removes near-constant predictors. The survivors go through $B{=}30$ bootstrap resamples of the training
set; each resample trains a small XGBoost ($\text{max\_depth}{=}4$, 50 rounds) and keeps its top $K{=}15$
features by gain. A feature's selection frequency is
"""
        )
        st.latex(r"C(j) = \sum_{b=1}^{B} \mathbf{1}(j \in S_b), \qquad S^{*} = \{\, j \mid C(j) \ge \tau B \,\}")
        st.markdown("with $\\tau{=}0.6$ — a feature must land in the top-15 in **at least 18 of 30** resamples. This narrows 111 candidates to **14 stable predictors**.")
        show_paper_figure(FIG_FEATURE_SELECTION, "Fig. 3 (paper) — the embedded + stability selection pipeline.")
        show_paper_figure(FIG_SELECTION_OUTCOME, "Fig. 4 (paper) — 14 of 111 features (12.6%) retained; 97 (87.4%) dropped.")

        st.markdown("**5. Final model.**")
        st.latex(r"\hat{y}_i^{(t)} = \sum_{k=1}^{t} f_k(x_i), \qquad \mathcal{L} = \sum_i (y_i-\hat y_i)^2 + \gamma T + \tfrac{1}{2}\lambda\sum_j w_j^2")
        st.markdown(
            "XGBoost trained on the 14 selected features, with 5-fold time-series cross-validation "
            "(`xgb.cv`, up to 1000 rounds, early stopping) picking the boosting-round count. "
            "Split gain, used to rank features throughout, is:"
        )
        st.latex(r"\mathrm{Gain}(j) = \tfrac12\!\left[\tfrac{G_L^2}{H_L+\lambda} + \tfrac{G_R^2}{H_R+\lambda} - \tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma")

        st.markdown("**6. Evaluation & interpretability.** MAE / MSE / RMSE / MAPE on the 2019-2020 test set, benchmarked against nine baselines (Transformer, TFT, Autoformer, DeepAR, DLinear, Deep RVFL, CNN-LSTM, CNN-ANN, ARIMA-LSTM) with Wilcoxon signed-rank and Friedman significance tests, plus SHAP for feature-level explanation.")
        show_paper_figure(FIG_SHAP_BEESWARM, "Fig. 10 (paper) — SHAP beeswarm over 2,000 test observations. This app's live run shows gain-based importance (Results & Validation); full SHAP is heavier to compute and shown here from the paper's own run.")

        st.write("")
        st.markdown("#### Model diagnostics (published run)")
        show_paper_figure(FIG_MODEL_DIAGNOSTICS, "Fig. 8 (paper) — (A) learning curve, (B) actual vs. predicted 2019-2020, (C) residual distribution, (D) top-10 gain importances.")
        if FEATURE_IMPORTANCE_IMG.exists():
            st.image(str(FEATURE_IMPORTANCE_IMG), use_container_width=True, caption="Feature importance for the selected features (project figure).")
        if WEATHER_PLOTS_IMG.exists():
            st.image(str(WEATHER_PLOTS_IMG), use_container_width=True, caption="Partial dependence & residuals vs. weather predictors (project figure).")

    with tab3:
        st.caption("Table 13 (paper) — the 14 features actually selected in the published run, ranked by gain.")
        feat_df = pd.DataFrame(SELECTED_FEATURES_REF, columns=["Feature", "Gain", "% of total gain", "Corr. with residual", "Avg. |SHAP|", "Selection frequency"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.caption("roll3_mean and lag_1 alone account for 45.3% + 9.4% ≈ 55% of total gain — recent load dynamics dominate, with weather (dew-point index, solar irradiance) and hour contributing smaller but consistent gains.")
    with tab4:
        st.markdown("".join(f'<span class="kw">{k}</span>' for k in PAPER["keywords"]), unsafe_allow_html=True)
    with tab5:
        st.markdown(
            f"""
This dashboard runs the real **ProphetBoost** research project of **Nazrul Amin**, Department of
Computer Engineering, Jeju National University, under the supervision of **Prof. Yung-Cheol Byun**.

**What's live:** the Prophet decomposition, feature engineering, bootstrap stability feature
selection, and final XGBoost model on **Train & Simulate** all run for real, on the actual
`totalload_new.csv` dataset, with your chosen hyperparameters — nothing there is precomputed.
**Results & Validation → Your run** is built entirely from that live run's output.

**What's a reference, not a live run:** the nine baselines under **Results & Validation → Published
benchmarks** (ARIMA-LSTM, DLinear, CNN-ANN, Deep RVFL, CNN-LSTM, Transformer, DeepAR, Autoformer, TFT)
require PyTorch training with Optuna hyperparameter search and take minutes to hours — those numbers,
along with the paper's own architecture figures above, are transcribed from the published paper
(`1-s2.0-S0045790626003526-main.pdf`), shown as a faithful reference rather than re-run here.

**Dataset:** ~9 years of hourly Jeju Island electricity load paired with hourly weather from four
stations (Jeju, Gosan, Sungsan, Seogwipo). Upload your own CSV via **Dataset → Upload new** to browse
or train the generic model on it — it's saved under `uploaded_datasets/` and persists across restarts.

{PAPER['conclusion']}
"""
        )
        st.markdown("---")
        st.markdown(f"**{PAPER['title']}**")
        st.markdown(f"*{PAPER['authors']}*")
        st.markdown(f"{PAPER['venue']} · {PAPER['citation']}")
        st.markdown(f"{PAPER['dates']}")
        st.markdown(f"[doi.org/{PAPER['doi']}](https://doi.org/{PAPER['doi']})")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
ROUTES = {
    "Overview": page_overview,
    "Dataset": page_dataset,
    "Train & Simulate": page_train_and_simulate,
    "Results & Validation": page_results_and_validation,
    "Research & About": page_research_and_about,
}
ROUTES[st.session_state.page]()
render_footer()
