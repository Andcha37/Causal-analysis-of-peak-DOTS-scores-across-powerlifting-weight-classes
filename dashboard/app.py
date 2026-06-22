from __future__ import annotations

from pathlib import Path
import math

import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Powerlifting Causal Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESULT_BOOK = OUTPUTS / "powerlifting_causal_analysis_results.xlsx"
DAG_PATH = Path(__file__).resolve().parent / "assets" / "causal_dag.svg"

SEX_LABEL = {"M": "남성", "F": "여성"}
SEX_COLOR = {"M": "#2563EB", "F": "#E11D74"}
MODEL_INFO = {
    "GPS-DR": {
        "title": "Generalized Propensity Score + Doubly Robust",
        "caption": "연속형 체중을 유지하고 RCS로 비선형 용량–반응을 추정합니다.",
    },
    "PSM": {
        "title": "Propensity Score Matching",
        "caption": "Q1–Q3의 무작위 체중 임계값으로 Heavy/Light를 정의해 1:1 매칭합니다.",
    },
    "CEM": {
        "title": "Coarsened Exact Matching",
        "caption": "공변량을 조대화해 동일 층 내 Heavy/Light 차이를 추정합니다.",
    },
    "Adjusted OLS": {
        "title": "Covariate-adjusted OLS",
        "caption": "체중 1kg당 Peak Dots 변화량을 통제변수와 함께 추정합니다.",
    },
}

CONFOUNDERS = {
    "초기 발전속도": ["Initial_Speed"],
    "경력 시작 나이": ["Career_Start_Age"],
    "경력 시작 연도": ["Career_Start_Year_Normalized"],
    "이전 출전횟수": ["Prior_Competition_Count"],
    "체급 변경횟수": ["Career_WeightClass_Change_Count"],
    "연맹 구분": ["IPFCategory_Non-IPF"],
    "개최 대륙": ["Continent_Asia_Oceania", "Continent_Europe_Africa", "Continent_Other"],
}


st.markdown(
    """
    <style>
    :root { --ink:#10233F; --muted:#607089; --blue:#2563EB; --cyan:#0EA5E9; --rose:#E11D74; }
    .stApp { background: linear-gradient(180deg, #F4F8FC 0%, #FFFFFF 38%); color: var(--ink); }
    [data-testid="stSidebar"] { background: #0B1F3A; }
    [data-testid="stSidebar"] * { color: #ECF5FF !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #10233F !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] input { color: #10233F !important; }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background: #1D4ED8; }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] * { color: #FFFFFF !important; }
    .block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1500px; }
    .hero {
        padding: 2.1rem 2.4rem; border-radius: 24px;
        background: radial-gradient(circle at 85% 10%, rgba(14,165,233,.34), transparent 28%),
                    linear-gradient(125deg, #071A33 0%, #123C72 62%, #0E7490 100%);
        box-shadow: 0 22px 50px rgba(15,45,85,.18); color: white; margin-bottom: 1.2rem;
    }
    .hero-kicker { font-size:.77rem; letter-spacing:.16em; text-transform:uppercase; color:#7DD3FC; font-weight:700; }
    .hero h1 { font-size:2.35rem; margin:.35rem 0 .5rem 0; line-height:1.12; }
    .hero p { max-width:850px; color:#D8EAFE; margin:0; font-size:1.02rem; }
    .hero-badge { display:inline-block; margin-top:1rem; padding:.38rem .72rem; border:1px solid rgba(255,255,255,.25); border-radius:999px; color:#E0F2FE; font-size:.78rem; }
    .section-kicker { color:#0284C7; font-size:.76rem; letter-spacing:.13em; font-weight:800; text-transform:uppercase; margin-bottom:.2rem; }
    .panel { background:#FFFFFF; border:1px solid #E3ECF5; border-radius:18px; padding:1.25rem 1.35rem; box-shadow:0 8px 26px rgba(32,72,112,.07); }
    .result-card { padding:1.15rem 1.25rem; border-radius:17px; background:#FFFFFF; border-left:5px solid var(--accent); box-shadow:0 7px 22px rgba(24,55,90,.08); min-height:128px; }
    .result-card .label { font-size:.75rem; text-transform:uppercase; letter-spacing:.09em; color:#64748B; font-weight:750; }
    .result-card .value { font-size:1.85rem; font-weight:800; color:#10233F; margin:.22rem 0; }
    .result-card .note { font-size:.82rem; color:#64748B; }
    .question-card { border-radius:18px; padding:1.2rem; background:linear-gradient(145deg,#EFF7FF,#FFFFFF); border:1px solid #D8EAFE; min-height:145px; }
    .question-card b { color:#075985; }
    .method-chip { display:inline-block; border-radius:999px; padding:.28rem .65rem; margin:.15rem .15rem .15rem 0; background:#E0F2FE; color:#075985; font-size:.75rem; font-weight:700; }
    .callout { border-left:4px solid #0EA5E9; background:#F0F9FF; padding:.9rem 1rem; border-radius:0 12px 12px 0; color:#164E63; }
    .warning-callout { border-left-color:#F59E0B; background:#FFFBEB; color:#78350F; }
    .takeaway { padding:1.05rem 1.15rem; border-radius:15px; background:#F8FAFC; border:1px solid #E2E8F0; margin-bottom:.65rem; }
    div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E4EDF6; padding:1rem 1.05rem; border-radius:16px; box-shadow:0 7px 22px rgba(24,55,90,.06); }
    div[data-testid="stMetric"] label { color:#64748B; }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; background:#EAF1F8; border-radius:14px; padding:.35rem; }
    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:.55rem .8rem; }
    .stTabs [aria-selected="true"] { background:white; box-shadow:0 3px 12px rgba(20,55,90,.10); }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    if not RESULT_BOOK.exists():
        raise FileNotFoundError(f"결과 파일이 없습니다: {RESULT_BOOK}")
    athletes = {
        sex: pd.read_csv(OUTPUTS / "derived" / f"athlete_level_{sex}.csv")
        for sex in ["M", "F"]
    }
    tables = {
        "overview": pd.read_excel(RESULT_BOOK, sheet_name="Athlete_Overview"),
        "ols": pd.read_excel(RESULT_BOOK, sheet_name="OLS_Key_Coefficients"),
        "mediation": pd.read_excel(RESULT_BOOK, sheet_name="Mediation"),
        "matching": pd.read_excel(RESULT_BOOK, sheet_name="Matching_Summary"),
        "gps_diag": pd.read_excel(RESULT_BOOK, sheet_name="GPS_Diagnostics"),
    }
    for sex in ["M", "F"]:
        tables[f"{sex}_match_balance"] = pd.read_excel(
            RESULT_BOOK, sheet_name=f"{sex}_Matching_Balance"
        ).rename(columns={"Unnamed: 0": "Covariate"})
        tables[f"{sex}_gps_balance"] = pd.read_excel(
            RESULT_BOOK, sheet_name=f"{sex}_GPS_Balance"
        )
        tables[f"{sex}_gps_curve"] = pd.read_csv(
            OUTPUTS / "tables" / f"{sex}_GPS_DRF.csv"
        )
        for method in ["PSM", "CEM"]:
            tables[f"{sex}_{method}_iterations"] = pd.read_csv(
                OUTPUTS / "tables" / f"{sex}_{method}_threshold_iterations.csv"
            )
    return athletes, tables


try:
    ATHLETES, TABLES = load_data()
except Exception as exc:
    st.error(f"대시보드 데이터를 불러오지 못했습니다: {exc}")
    st.info("메인 분석 노트북을 끝까지 실행해 outputs 폴더를 먼저 생성하세요.")
    st.stop()


def selected_columns(labels: list[str], df: pd.DataFrame) -> list[str]:
    columns = []
    for label in labels:
        columns.extend(CONFOUNDERS[label])
    return [column for column in columns if column in df.columns]


def format_effect(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.{digits}f}"


def html_result_card(label: str, value: str, note: str, color: str):
    st.markdown(
        f"""<div class="result-card" style="--accent:{color}">
        <div class="label">{label}</div><div class="value">{value}</div>
        <div class="note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def filter_athletes(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    return df[df["WeightClass_Bin"].astype(str).isin(classes)].copy() if classes else df.copy()


def ols_effect(df: pd.DataFrame, confounders: list[str]):
    cols = ["Peak_Dots", "Avg_Bodyweight"] + confounders
    work = df[cols].dropna().astype(float)
    if len(work) < max(50, len(confounders) * 8):
        return None
    X = sm.add_constant(work[["Avg_Bodyweight"] + confounders], has_constant="add")
    fit = sm.OLS(work["Peak_Dots"], X).fit(cov_type="HC3")
    ci = fit.conf_int().loc["Avg_Bodyweight"]
    return {
        "effect": fit.params["Avg_Bodyweight"],
        "lower": ci.iloc[0],
        "upper": ci.iloc[1],
        "pvalue": fit.pvalues["Avg_Bodyweight"],
        "n": len(work),
        "r2": fit.rsquared_adj,
    }


@st.cache_data(show_spinner=False)
def propensity_proxy(df: pd.DataFrame, confounders: tuple[str, ...]):
    columns = ["Avg_Bodyweight"] + list(confounders)
    work = df[columns].dropna().copy()
    work["Heavy"] = (work["Avg_Bodyweight"] > work["Avg_Bodyweight"].median()).astype(int)
    X = work[list(confounders)].astype(float)
    if X.shape[1] == 0:
        work["Propensity"] = work["Heavy"].mean()
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=42))
        model.fit(X, work["Heavy"])
        work["Propensity"] = model.predict_proba(X)[:, 1]
    work["Group"] = work["Heavy"].map({0: "Light", 1: "Heavy"})
    return work[["Propensity", "Group", "Heavy"]]


def balance_long(sex: str, model_name: str):
    if model_name == "GPS-DR":
        data = TABLES[f"{sex}_gps_balance"].rename(
            columns={"Unadjusted_Abs_Corr": "Before", "Weighted_Abs_Corr": "After"}
        )
        value_name = "Absolute correlation"
    else:
        raw = TABLES[f"{sex}_match_balance"]
        after = model_name if model_name in ["PSM", "CEM"] else "PSM"
        data = raw[["Covariate", "Unmatched", after]].rename(
            columns={"Unmatched": "Before", after: "After"}
        )
        value_name = "Absolute SMD"
    long = data.melt("Covariate", var_name="Stage", value_name="Balance")
    return long, value_name


def within_class_slopes(df: pd.DataFrame, confounders: list[str]):
    rows = []
    for weight_class, group in df.groupby("WeightClass_Bin", observed=True):
        result = ols_effect(group, confounders)
        if result:
            rows.append({"Weight class": str(weight_class), **result})
    return pd.DataFrame(rows)


def nested_specifications(df: pd.DataFrame):
    specs = {
        "Unadjusted": [],
        "+ Demographics": ["Career_Start_Age", "Career_Start_Year_Normalized"],
        "+ Career": ["Career_Start_Age", "Career_Start_Year_Normalized", "Prior_Competition_Count",
                      "Career_WeightClass_Change_Count"],
        "+ Full": ["Initial_Speed", "Career_Start_Age", "Career_Start_Year_Normalized",
                   "Prior_Competition_Count", "Career_WeightClass_Change_Count",
                   "IPFCategory_Non-IPF", "Continent_Asia_Oceania", "Continent_Europe_Africa", "Continent_Other"],
    }
    rows = []
    for name, columns in specs.items():
        columns = [c for c in columns if c in df.columns]
        result = ols_effect(df, columns)
        if result:
            rows.append({"Specification": name, **result})
    return pd.DataFrame(rows)


def dag_html():
    return """
    <svg viewBox="0 0 1000 430" width="100%" height="430" role="img" aria-label="Causal DAG"
         style="display:block;background:#fff;font-family:Inter,Segoe UI,sans-serif;border-radius:22px">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#7890AA"/></marker>
        <filter id="shadow"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-opacity=".13"/></filter>
      </defs>
      <rect width="1000" height="430" rx="22" fill="#F7FAFD"/>
      <text x="42" y="42" font-size="15" fill="#55708D" font-weight="700" letter-spacing="2">IDENTIFICATION DAG</text>
      <g stroke="#7890AA" stroke-width="2.5" fill="none" marker-end="url(#arrow)">
        <path d="M250 120 C380 110, 410 140, 500 187"/>
        <path d="M250 220 C360 220, 420 210, 500 210"/>
        <path d="M250 320 C380 315, 420 260, 500 228"/>
        <path d="M640 210 C720 210, 755 210, 810 210"/>
        <path d="M250 120 C470 45, 720 70, 810 178"/>
        <path d="M250 220 C500 135, 700 140, 810 190"/>
        <path d="M250 320 C520 360, 720 315, 810 238"/>
        <path d="M566 250 C565 285, 565 302, 565 326"/>
        <path d="M642 350 C745 345, 790 285, 835 247"/>
      </g>
      <g filter="url(#shadow)">
        <g><rect x="55" y="80" width="195" height="72" rx="16" fill="#E0F2FE" stroke="#7DD3FC"/>
          <text x="152" y="109" text-anchor="middle" font-size="15" font-weight="700" fill="#075985">Athlete background</text>
          <text x="152" y="132" text-anchor="middle" font-size="12" fill="#39708C">age · career start · year</text></g>
        <g><rect x="55" y="184" width="195" height="72" rx="16" fill="#F3E8FF" stroke="#D8B4FE"/>
          <text x="152" y="213" text-anchor="middle" font-size="15" font-weight="700" fill="#6B21A8">Career history</text>
          <text x="152" y="236" text-anchor="middle" font-size="12" fill="#8053A3">appearances · class changes</text></g>
        <g><rect x="55" y="288" width="195" height="72" rx="16" fill="#ECFCCB" stroke="#BEF264"/>
          <text x="152" y="317" text-anchor="middle" font-size="15" font-weight="700" fill="#3F6212">Context</text>
          <text x="152" y="340" text-anchor="middle" font-size="12" fill="#5D7738">federation · continent</text></g>
        <g><rect x="500" y="174" width="142" height="78" rx="20" fill="#DBEAFE" stroke="#60A5FA" stroke-width="2"/>
          <text x="571" y="206" text-anchor="middle" font-size="17" font-weight="800" fill="#1D4ED8">Body weight</text>
          <text x="571" y="231" text-anchor="middle" font-size="12" fill="#3B65A3">Treatment</text></g>
        <g><rect x="802" y="174" width="155" height="78" rx="20" fill="#FFE4E6" stroke="#FDA4AF" stroke-width="2"/>
          <text x="879" y="206" text-anchor="middle" font-size="17" font-weight="800" fill="#BE123C">Peak Dots</text>
          <text x="879" y="231" text-anchor="middle" font-size="12" fill="#A6415C">Outcome</text></g>
        <g><rect x="500" y="326" width="142" height="62" rx="16" fill="#FEF3C7" stroke="#FCD34D"/>
          <text x="571" y="353" text-anchor="middle" font-size="14" font-weight="750" fill="#92400E">Initial speed</text>
          <text x="571" y="374" text-anchor="middle" font-size="11" fill="#A46624">Mediator candidate</text></g>
      </g>
      <text x="42" y="409" font-size="12" fill="#7890AA">Arrows encode the assumed adjustment structure—not proof of causality.</text>
    </svg>
    """


# Sidebar
with st.sidebar:
    st.markdown("### ⚡ Causal Lab")
    st.caption("Presentation & portfolio dashboard")
    st.divider()
    population = st.radio("분석 집단", ["남성", "여성", "남녀 비교"], index=2)
    selected_sexes = ["M", "F"] if population == "남녀 비교" else (["M"] if population == "남성" else ["F"])
    primary_sex = selected_sexes[0]
    model_name = st.selectbox("인과 추정 모델", list(MODEL_INFO), index=0)

    all_classes = sorted(
        set().union(*[set(ATHLETES[s]["WeightClass_Bin"].astype(str).unique()) for s in selected_sexes]),
        key=lambda x: float(x.split("-")[0]),
    )
    chosen_classes = st.multiselect("체급 필터 (탐색 패널)", all_classes, default=all_classes)
    chosen_confounders = st.multiselect(
        "실시간 탐색 통제변수",
        list(CONFOUNDERS),
        default=list(CONFOUNDERS),
    )
    st.caption("체급·통제변수 필터는 실시간 탐색용입니다. 논문형 인과추정치는 전체 표본의 사전 계산 결과를 유지합니다.")
    st.divider()
    st.markdown("**Run profile**")
    st.write("Seed `42` · OLS `1,000×`")
    st.write("Matching `100×` · GPS `100×`")


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">OpenPowerlifting · Causal inference portfolio</div>
      <h1>Body Weight → Peak Performance</h1>
      <p>체중이 최고 Dots에 미치는 영향을 남녀별로 분리하고, 회귀·매칭·연속형 GPS-DR로 교차 검증합니다.</p>
      <span class="hero-badge">3.7M raw records → 5,984 athlete-level trajectories</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["01 Overview", "02 Data & DAG", "03 Causal Model", "04 Results", "05 Robustness", "06 Interpretation"])


with tabs[0]:
    st.markdown('<div class="section-kicker">Problem definition</div>', unsafe_allow_html=True)
    st.subheader("상관관계가 아니라, 체중 변화의 조건부 인과효과를 묻습니다")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown('<div class="question-card"><b>Primary question</b><br><br>평균 체중이 증가할 때 선수의 생애 최고 Dots는 어떻게 변하는가?</div>', unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="question-card"><b>Heterogeneity</b><br><br>그 효과의 방향과 크기는 남성과 여성에게 동일한가?</div>', unsafe_allow_html=True)
    with q3:
        st.markdown('<div class="question-card"><b>Mechanism</b><br><br>초기 발전속도가 체중과 최고 성과 사이를 매개하는가?</div>', unsafe_allow_html=True)

    st.markdown("### Study at a glance")
    metric_cols = st.columns(4)
    selected_df = pd.concat([filter_athletes(ATHLETES[s], chosen_classes).assign(Sex=s) for s in selected_sexes])
    with metric_cols[0]:
        st.metric("분석 선수", f"{len(selected_df):,}", "탐색 필터 적용")
    with metric_cols[1]:
        st.metric("평균 Peak Dots", f"{selected_df['Peak_Dots'].mean():.1f}")
    with metric_cols[2]:
        st.metric("평균 체중", f"{selected_df['Avg_Bodyweight'].mean():.1f} kg")
    with metric_cols[3]:
        efficiency = (selected_df["Peak_Dots"] / selected_df["Avg_Bodyweight"]).mean()
        st.metric("상대 효율 지표", f"{efficiency:.2f}", "Peak Dots / kg · 기술통계")

    st.markdown("### Headline causal estimates")
    estimate_cols = st.columns(len(selected_sexes))
    for col, sex in zip(estimate_cols, selected_sexes):
        with col:
            if model_name == "Adjusted OLS":
                sex_label = SEX_LABEL[sex]
                row = TABLES["ols"].loc[
                    TABLES["ols"]["성별"].eq(sex_label)
                    & TABLES["ols"]["모형"].eq("E1")
                    & TABLES["ols"]["변수"].eq("Avg_Bodyweight")
                ].iloc[0]
                value = format_effect(row["계수"], 3)
                note = f"Dots / +1kg · 95% CI [{row['BS 95% 하한']:.3f}, {row['BS 95% 상한']:.3f}]"
            elif model_name in ["PSM", "CEM"]:
                row = TABLES["matching"].query("Sex == @sex and Method == @model_name").iloc[0]
                value = format_effect(row["Mean_Effect"], 2)
                note = f"Heavy vs Light · 95% CI [{row['CI_Lower']:.2f}, {row['CI_Upper']:.2f}]"
            else:
                curve = TABLES[f"{sex}_gps_curve"]
                value = format_effect(curve["Estimated_Peak_Dots"].iloc[-1] - curve["Estimated_Peak_Dots"].iloc[0], 1)
                note = "1–99% 체중 범위의 예측 곡선 종단 차이"
            html_result_card(f"{SEX_LABEL[sex]} · {model_name}", value, note, SEX_COLOR[sex])

    st.markdown(
        '<div class="callout"><b>핵심 관찰:</b> 남성은 체중과 Peak Dots의 양의 방향, 여성은 음의 방향이 반복적으로 나타납니다. 다만 남성 CEM은 신뢰구간이 0을 포함해 방법 선택에 더 민감합니다.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Evidence pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    for col, title, text in [
        (p1, "01 Clean", "Raw · tested · sanctioned"),
        (p2, "02 Trace", "30-day interpolation + EMA"),
        (p3, "03 Adjust", "Age · career · context"),
        (p4, "04 Estimate", "OLS · PSM · CEM · GPS-DR"),
        (p5, "05 Stress-test", "Bootstrap · thresholds · balance"),
    ]:
        with col:
            st.markdown(f'<div class="panel"><b>{title}</b><br><span style="color:#64748B;font-size:.82rem">{text}</span></div>', unsafe_allow_html=True)


with tabs[1]:
    st.markdown('<div class="section-kicker">Data & design</div>', unsafe_allow_html=True)
    st.subheader("Treatment, outcome, confounders—and the assumptions connecting them")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("#### Variable map")
        variable_map = pd.DataFrame([
            ["Treatment", "Avg_Bodyweight", "선택 체급 내 평균 체중(kg)"],
            ["Outcome", "Peak_Dots", "EMA 생애주기 곡선의 최고점"],
            ["Mediator", "Initial_Speed", "첫 고성장 구간의 연간 Dots 기울기"],
            ["Confounder", "Career_Start_Age / Year", "경력 시작 시점"],
            ["Confounder", "Prior competitions", "이전 출전 경험"],
            ["Confounder", "Class changes", "체급 이동 이력"],
            ["Context", "Federation / Continent", "연맹·개최 지역"],
        ], columns=["Role", "Variable", "Operational definition"])
        st.dataframe(variable_map, hide_index=True, width="stretch")
        st.markdown("#### Cohort construction")
        st.markdown(
            """
            <span class="method-chip">Raw equipment</span><span class="method-chip">Drug-tested</span>
            <span class="method-chip">Sanctioned</span><span class="method-chip">Age ≥ 18</span>
            <span class="method-chip">≥ 6 meets / class</span><span class="method-chip">Career ≥ 180 days</span>
            """, unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Causal graph (DAG)")
        st.image(str(DAG_PATH), width="stretch")

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown('<div class="panel"><b>Exchangeability</b><br><span style="color:#64748B">측정된 경력·연맹·지역 변수를 조건부로 했을 때 잔여 교란이 작다고 가정합니다.</span></div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="panel"><b>Positivity</b><br><span style="color:#64748B">공변량 조합별로 충분한 체중 변이가 존재해야 합니다. overlap과 ESS로 점검합니다.</span></div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="panel"><b>Consistency</b><br><span style="color:#64748B">평균 체중이라는 처치가 선수마다 비교 가능한 개입 의미를 가진다고 가정합니다.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="callout warning-callout"><b>중요:</b> 훈련, 영양, 신장, 골격근량, 체지방률은 데이터에 없어 잔여 교란 가능성이 있습니다. DAG는 식별 가정을 명시하는 도구이지 인과성을 증명하는 그림이 아닙니다.</div>', unsafe_allow_html=True)


with tabs[2]:
    st.markdown('<div class="section-kicker">Causal model</div>', unsafe_allow_html=True)
    st.subheader(MODEL_INFO[model_name]["title"])
    st.caption(MODEL_INFO[model_name]["caption"])
    active_df = ATHLETES[primary_sex]
    active_cols = selected_columns(chosen_confounders, active_df)

    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        st.markdown("#### Treatment overlap proxy")
        overlap = propensity_proxy(active_df, tuple(active_cols))
        hist = alt.Chart(overlap).mark_bar(opacity=.58, binSpacing=0).encode(
            x=alt.X("Propensity:Q", bin=alt.Bin(maxbins=35), title="P(Heavy | selected covariates)"),
            y=alt.Y("count():Q", stack=None, title="Athletes"),
            color=alt.Color("Group:N", scale=alt.Scale(domain=["Light", "Heavy"], range=["#0EA5E9", "#E11D74"])),
            tooltip=["Group:N", "count():Q"],
        ).properties(height=340)
        st.altair_chart(hist, width="stretch")
        extreme = ((overlap["Propensity"] < .05) | (overlap["Propensity"] > .95)).mean()
        st.caption(f"이진 Heavy/Light overlap 진단용 proxy · 극단 성향점수 비율 {extreme:.1%}")
    with c2:
        st.markdown("#### Covariate balance")
        if model_name == "Adjusted OLS":
            st.info("OLS는 가중·매칭 균형을 만들지 않습니다. 모델 명세 민감도는 Robustness 탭에서 확인하세요.")
            balance_model = "GPS-DR"
        else:
            balance_model = model_name
        balance, balance_title = balance_long(primary_sex, balance_model)
        balance_chart = alt.Chart(balance).mark_circle(size=95, opacity=.82).encode(
            x=alt.X("Balance:Q", title=balance_title),
            y=alt.Y("Covariate:N", sort="-x", title=None),
            color=alt.Color("Stage:N", scale=alt.Scale(domain=["Before", "After"], range=["#94A3B8", SEX_COLOR[primary_sex]])),
            shape="Stage:N",
            tooltip=["Covariate", "Stage", alt.Tooltip("Balance", format=".3f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [.1]})).mark_rule(strokeDash=[5, 4], color="#F59E0B").encode(x="x:Q")
        st.altair_chart((balance_chart + rule).properties(height=350), width="stretch")

    diag = TABLES["gps_diag"].loc[
        TABLES["gps_diag"]["성별"].eq(SEX_LABEL[primary_sex])
    ].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("GPS effective N", f"{diag['Effective_Sample_Size']:,.0f}")
    d2.metric("ESS retention", f"{diag['Effective_Sample_Size']/diag['N']:.1%}")
    d3.metric("Max stabilized weight", f"{diag['Weight_Max']:.2f}")
    d4.metric("Valid bootstrap", f"{int(diag['Valid_Bootstrap'])}/100")

    st.markdown("#### Active adjustment set")
    if chosen_confounders:
        st.markdown("".join(f'<span class="method-chip">{item}</span>' for item in chosen_confounders), unsafe_allow_html=True)
    else:
        st.warning("통제변수가 선택되지 않았습니다. 이 상태의 실시간 OLS는 기술적 연관에 가깝습니다.")


with tabs[3]:
    st.markdown('<div class="section-kicker">Estimation results</div>', unsafe_allow_html=True)
    st.subheader("체중에 따라 인과효과가 어떻게 움직이는가")

    st.markdown("#### GPS-DR dose–response")
    curve_frames = []
    for sex in selected_sexes:
        frame = TABLES[f"{sex}_gps_curve"].copy()
        frame["Sex"] = SEX_LABEL[sex]
        curve_frames.append(frame)
    curves = pd.concat(curve_frames, ignore_index=True)
    band = alt.Chart(curves).mark_area(opacity=.14).encode(
        x=alt.X("Bodyweight:Q", title="Average body weight (kg)"),
        y=alt.Y("CI_Lower:Q", title="Estimated Peak Dots"),
        y2="CI_Upper:Q",
        color=alt.Color("Sex:N", scale=alt.Scale(domain=["남성", "여성"], range=[SEX_COLOR["M"], SEX_COLOR["F"]])),
    )
    line = alt.Chart(curves).mark_line(strokeWidth=3).encode(
        x="Bodyweight:Q", y="Estimated_Peak_Dots:Q", color="Sex:N",
        tooltip=["Sex", alt.Tooltip("Bodyweight", format=".1f"), alt.Tooltip("Estimated_Peak_Dots", format=".1f"),
                 alt.Tooltip("CI_Lower", format=".1f"), alt.Tooltip("CI_Upper", format=".1f")],
    )
    st.altair_chart((band + line).properties(height=430), width="stretch")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### Descriptive distribution")
        sample = selected_df.sample(min(3_000, len(selected_df)), random_state=42)
        sample["Sex label"] = sample["Sex"].map(SEX_LABEL)
        points = alt.Chart(sample).mark_circle(opacity=.22, size=28).encode(
            x=alt.X("Avg_Bodyweight:Q", title="Average body weight (kg)"),
            y=alt.Y("Peak_Dots:Q", title="Peak Dots"),
            color=alt.Color("Sex label:N", scale=alt.Scale(domain=["남성", "여성"], range=[SEX_COLOR["M"], SEX_COLOR["F"]])),
            tooltip=["Name", "WeightClass_Bin", alt.Tooltip("Avg_Bodyweight", format=".1f"), alt.Tooltip("Peak_Dots", format=".1f")],
        )
        regression = points.transform_regression("Avg_Bodyweight", "Peak_Dots", groupby=["Sex label"]).mark_line(strokeWidth=3)
        st.altair_chart((points + regression).properties(height=370), width="stretch")
    with right:
        st.markdown("#### Weight-class ranking")
        ranking = (selected_df.groupby(["Sex", "WeightClass_Bin"], observed=True)
                   .agg(Athletes=("Name", "size"), Mean_Peak_Dots=("Peak_Dots", "mean"),
                        Mean_Initial_Speed=("Initial_Speed", "mean"), Mean_Bodyweight=("Avg_Bodyweight", "mean"))
                   .reset_index())
        ranking["Sex"] = ranking["Sex"].map(SEX_LABEL)
        ranking["Efficiency"] = ranking["Mean_Peak_Dots"] / ranking["Mean_Bodyweight"]
        ranking = ranking.sort_values("Mean_Peak_Dots", ascending=False)
        st.dataframe(
            ranking.rename(columns={"Sex": "성별", "WeightClass_Bin": "체급", "Athletes": "N",
                                    "Mean_Peak_Dots": "평균 Peak Dots", "Mean_Initial_Speed": "초기 발전속도",
                                    "Efficiency": "Dots/kg"})
            [["성별", "체급", "N", "평균 Peak Dots", "초기 발전속도", "Dots/kg"]]
            .style.format({"평균 Peak Dots": "{:.1f}", "초기 발전속도": "{:.1f}", "Dots/kg": "{:.2f}"}),
            hide_index=True, width="stretch", height=370,
        )

    st.markdown("#### Exploratory within-class effect heterogeneity")
    st.caption("아래 계수는 선택한 통제변수로 체급 내부에서 다시 적합한 탐색적 OLS 기울기입니다. 사전 정의된 CATE/ATE로 해석하지 않습니다.")
    cate_frames = []
    for sex in selected_sexes:
        filtered = filter_athletes(ATHLETES[sex], chosen_classes)
        slopes = within_class_slopes(filtered, selected_columns(chosen_confounders, filtered))
        if not slopes.empty:
            slopes["Sex"] = SEX_LABEL[sex]
            cate_frames.append(slopes)
    if cate_frames:
        cate = pd.concat(cate_frames)
        error = alt.Chart(cate).mark_rule(strokeWidth=2).encode(
            x=alt.X("lower:Q", title="Adjusted within-class slope (Dots / kg)"), x2="upper:Q",
            y=alt.Y("Weight class:N", sort=all_classes, title="Weight class"), color="Sex:N"
        )
        dots = alt.Chart(cate).mark_point(filled=True, size=90).encode(
            x="effect:Q", y="Weight class:N", color=alt.Color("Sex:N", scale=alt.Scale(range=[SEX_COLOR["M"], SEX_COLOR["F"]])),
            tooltip=["Sex", "Weight class", alt.Tooltip("effect", format=".3f"), alt.Tooltip("lower", format=".3f"), alt.Tooltip("upper", format=".3f"), "n:Q"]
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#64748B", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((error + dots + zero).properties(height=370), width="stretch")


with tabs[4]:
    st.markdown('<div class="section-kicker">Robustness & sensitivity</div>', unsafe_allow_html=True)
    st.subheader("하나의 모델이 아니라, 결과가 얼마나 버티는지를 봅니다")

    st.markdown("#### Method triangulation")
    method_rows = []
    for sex in selected_sexes:
        sex_label = SEX_LABEL[sex]
        ols_row = TABLES["ols"].loc[
            TABLES["ols"]["성별"].eq(sex_label)
            & TABLES["ols"]["모형"].eq("E1")
            & TABLES["ols"]["변수"].eq("Avg_Bodyweight")
        ].iloc[0]
        method_rows.append({"Sex": SEX_LABEL[sex], "Method": "Adjusted OLS", "Estimate": ols_row["계수"],
                            "Lower": ols_row["BS 95% 하한"], "Upper": ols_row["BS 95% 상한"], "Estimand": "Dots / +1kg"})
        for method in ["PSM", "CEM"]:
            row = TABLES["matching"].query("Sex == @sex and Method == @method").iloc[0]
            method_rows.append({"Sex": SEX_LABEL[sex], "Method": method, "Estimate": row["Mean_Effect"],
                                "Lower": row["CI_Lower"], "Upper": row["CI_Upper"], "Estimand": "Heavy − Light"})
    method_table = pd.DataFrame(method_rows)
    st.dataframe(method_table.style.format({"Estimate": "{:+.3f}", "Lower": "{:+.3f}", "Upper": "{:+.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("OLS는 1kg당 효과, PSM/CEM은 Heavy–Light 효과이므로 크기를 직접 비교하지 않고 방향과 유의성만 삼각검증합니다.")

    st.markdown("#### Threshold sensitivity")
    threshold_frames = []
    for sex in selected_sexes:
        for method in ["PSM", "CEM"]:
            frame = TABLES[f"{sex}_{method}_iterations"].copy()
            frame["Sex"] = SEX_LABEL[sex]
            frame["Method"] = method
            threshold_frames.append(frame)
    threshold = pd.concat(threshold_frames)
    threshold_chart = alt.Chart(threshold).mark_circle(size=45, opacity=.58).encode(
        x=alt.X("cutoff:Q", title="Random body-weight cutoff (kg)"),
        y=alt.Y("effect:Q", title="Estimated Heavy − Light effect"),
        color=alt.Color("Sex:N", scale=alt.Scale(domain=["남성", "여성"], range=[SEX_COLOR["M"], SEX_COLOR["F"]])),
        shape="Method:N",
        tooltip=["Sex", "Method", alt.Tooltip("cutoff", format=".1f"), alt.Tooltip("effect", format=".2f"), "matched_n:Q"],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#64748B", strokeDash=[5, 4]).encode(y="y:Q")
    st.altair_chart((threshold_chart + zero).properties(height=380), width="stretch")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### Confounder specification sensitivity")
        specs = nested_specifications(ATHLETES[primary_sex])
        bars = alt.Chart(specs).mark_rule(strokeWidth=3).encode(
            x=alt.X("lower:Q", title="Body-weight coefficient (Dots / kg)"), x2="upper:Q",
            y=alt.Y("Specification:N", sort=list(specs["Specification"]), title=None),
        )
        points = alt.Chart(specs).mark_point(size=120, filled=True, color=SEX_COLOR[primary_sex]).encode(
            x="effect:Q", y="Specification:N",
            tooltip=["Specification", alt.Tooltip("effect", format=".3f"), alt.Tooltip("lower", format=".3f"), alt.Tooltip("upper", format=".3f"), alt.Tooltip("r2", format=".3f")],
        )
        zero_x = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#94A3B8", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((bars + points + zero_x).properties(height=290), width="stretch")
    with right:
        st.markdown("#### Mediation sensitivity")
        selected_labels = [SEX_LABEL[s] for s in selected_sexes]
        mediation = TABLES["mediation"].loc[
            TABLES["mediation"]["성별"].isin(selected_labels)
        ].copy()
        for _, row in mediation.iterrows():
            color = SEX_COLOR["M" if row["성별"] == "남성" else "F"]
            html_result_card(
                f"{row['성별']} · Indirect effect",
                format_effect(row["Indirect_Effect_Mean"], 4),
                f"95% CI [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}] · 0 포함",
                color,
            )
        st.markdown('<div class="callout warning-callout"><b>Not estimated:</b> Rosenbaum bounds와 negative-control 분석은 현재 산출물에 없습니다. 임계값·명세·모델 삼각검증을 현재의 민감도 근거로 사용합니다.</div>', unsafe_allow_html=True)


with tabs[5]:
    st.markdown('<div class="section-kicker">Interpretation</div>', unsafe_allow_html=True)
    st.subheader("결과를 체급 전략으로 번역하되, 데이터가 말하지 않은 것까지 말하지 않습니다")

    male, female = st.columns(2, gap="large")
    with male:
        st.markdown(
            """<div class="result-card" style="--accent:#2563EB">
            <div class="label">MEN · CONSISTENT DIRECTION, METHOD SENSITIVITY</div>
            <div class="value">Weight ↑ → Peak Dots ↑</div>
            <div class="note">OLS와 PSM은 양의 효과. CEM CI는 0을 포함해 강한 단정은 피해야 합니다.</div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - 체중 증가와 함께 절대근력 증가가 Dots 패널티를 상쇄했을 가능성
            - 무조건적인 벌크업보다 근육량 중심의 체중 증가라는 조건부 해석
            - CEM 불확실성 때문에 “항상 이득”이 아니라 평균적 경향으로 표현
            """
        )
    with female:
        st.markdown(
            """<div class="result-card" style="--accent:#E11D74">
            <div class="label">WOMEN · ROBUST NEGATIVE DIRECTION</div>
            <div class="value">Weight ↑ → Peak Dots ↓</div>
            <div class="note">OLS·PSM·CEM에서 음의 방향. 체중 대비 근력 효율 관리가 더 중요할 수 있습니다.</div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - 체중 증가가 Dots 보정 패널티를 충분히 상쇄하지 못했을 가능성
            - 체지방 중심 증량보다 lean mass와 기술 효율에 초점
            - 개인별 훈련 상태와 체성분이 없어 생리학적 원인을 직접 검증한 것은 아님
            """
        )

    st.markdown("### Presentation-ready takeaways")
    for number, title, body in [
        ("01", "Sex is an effect modifier", "성별을 단순 통제변수로 넣는 대신 분석을 분리해야 정반대 방향이 드러납니다."),
        ("02", "The mediator hypothesis was not supported", "초기 발전속도의 간접효과 신뢰구간은 남녀 모두 0을 포함했습니다."),
        ("03", "Balance is part of the result", "추정치만이 아니라 SMD·가중 상관·ESS·overlap이 인과 주장 강도를 결정합니다."),
        ("04", "One estimator is not enough", "남성 CEM의 불확실성처럼 모델 간 불일치 자체가 중요한 분석 결과입니다."),
    ]:
        st.markdown(f'<div class="takeaway"><b style="color:#0284C7">{number}</b> &nbsp; <b>{title}</b><br><span style="color:#64748B">{body}</span></div>', unsafe_allow_html=True)

    st.markdown("### Limitations")
    limitations = pd.DataFrame([
        ["Unmeasured confounding", "훈련, 영양, 신장, 체성분 부재"],
        ["Treatment definition", "시간가변 체중을 선택 체급 내 평균으로 요약"],
        ["Selection", "6회 이상 출전하고 최고점이 관찰된 생존 표본"],
        ["Estimand mismatch", "OLS per-kg와 matching Heavy–Light는 서로 다른 효과"],
        ["External validity", "IPF·서구권 비중이 높아 전체 선수군 일반화에 제약"],
    ], columns=["Risk", "Implication"])
    st.dataframe(limitations, hide_index=True, width="stretch")

    st.markdown(
        '<div class="callout"><b>Bottom line:</b> 이 프로젝트의 강점은 “인과효과를 확정했다”가 아니라, 동일 질문을 여러 추정량과 진단으로 압박해 어떤 결론이 견고하고 어디서 불확실성이 남는지를 보여준다는 점입니다.</div>',
        unsafe_allow_html=True,
    )


st.markdown("---")
st.caption("Powerlifting Causal Lab · OpenPowerlifting public data · Observational causal inference portfolio")
