"""
Bank Customer Churn Intelligence – Streamlit dashboard
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from churn_utils import RAW_INPUT_COLS, risk_tier

ROOT = Path(__file__).parent
st.set_page_config(page_title="Churn Intelligence", page_icon="🏦", layout="wide")

NAVY, RED, AMBER, GREEN, GREY = "#1F3A5F", "#C0392B", "#E9A23B", "#2A9D8F", "#9AA5B1"
TIER_COLOR = {"High": RED, "Medium": AMBER, "Low": GREEN}

# ----------------------------------------------------------------------------- loaders
@st.cache_resource
def load_artifacts():
    art = joblib.load(ROOT / "models" / "churn_model.joblib")
    X_tr, X_te, y_tr, y_te = joblib.load(ROOT / "models" / "split.joblib")
    metrics = json.load(open(ROOT / "outputs" / "metrics.json"))
    return art, X_te, y_te, metrics


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "European_Bank.csv")


art, X_te, y_te, METRICS = load_artifacts()
PIPE, DEFAULT_THR, CHAMP = art["pipeline"], art["threshold"], art["name"]


def predict(df: pd.DataFrame) -> np.ndarray:
    return PIPE.predict_proba(df[RAW_INPUT_COLS])[:, 1]


def contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer SHAP-style contributions (log-odds) using XGBoost's native TreeSHAP."""
    Xt = PIPE[:-1].transform(df[RAW_INPUT_COLS])
    booster = PIPE[-1].get_booster()
    c = booster.predict(xgb.DMatrix(Xt), pred_contribs=True)
    out = pd.DataFrame(c[:, :-1], columns=Xt.columns, index=df.index)
    out["_bias"] = c[:, -1]
    return out, Xt


@st.cache_data
def test_scores():
    p = predict(X_te)
    return pd.DataFrame({"p": p, "y": y_te.values, **X_te.reset_index(drop=True).to_dict("list")})


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🏦 Customer profile")
st.sidebar.caption("Describe a customer; every tab that scores a customer uses this profile.")
geo = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"], index=1)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 18, 92, 45)
tenure = st.sidebar.slider("Tenure (years with bank)", 0, 10, 5)
credit = st.sidebar.slider("Credit score", 300, 850, 650)
balance = st.sidebar.number_input("Account balance (€)", 0.0, 260000.0, 120000.0, step=5000.0)
salary = st.sidebar.number_input("Estimated salary (€)", 0.0, 200000.0, 100000.0, step=5000.0)
nprod = st.sidebar.select_slider("Number of products", [1, 2, 3, 4], value=1)
card = st.sidebar.toggle("Has credit card", value=True)
active = st.sidebar.toggle("Active member", value=False)
st.sidebar.divider()
thr = st.sidebar.slider("Decision threshold", 0.05, 0.95, float(round(DEFAULT_THR, 2)), 0.01,
                        help="Customers with probability ≥ threshold are flagged as likely churners. "
                             "Default = F1-optimal threshold learned from cross-validation.")
BASE = pd.DataFrame([dict(CreditScore=credit, Geography=geo, Gender=gender, Age=age, Tenure=tenure,
                          Balance=float(balance), NumOfProducts=int(nprod), HasCrCard=int(card),
                          IsActiveMember=int(active), EstimatedSalary=float(salary))])

# ----------------------------------------------------------------------------- header
st.title("Bank Customer Churn Intelligence")
st.caption(f"Champion model: **{CHAMP}** · trained on 8,000 customers · held-out test ROC-AUC "
           f"**{METRICS['test'][CHAMP]['roc_auc']:.3f}** · decision threshold {thr:.2f}")

tabs = st.tabs(["🎯 Risk calculator", "📈 Probability distribution", "🔍 Feature importance",
                "🧪 What-if simulator", "📊 Model performance", "📦 Batch scoring"])


def tier_badge(p):
    t = risk_tier(p)
    return f"<span style='background:{TIER_COLOR[t]};color:white;padding:4px 14px;border-radius:14px;font-weight:600'>{t} risk</span>"


def gauge(p):
    st.markdown(
        f"""<div style='background:#E8ECF1;border-radius:8px;height:26px;position:relative;overflow:hidden'>
        <div style='background:linear-gradient(90deg,{GREEN} 0%,{AMBER} 45%,{RED} 100%);width:100%;height:100%;opacity:.35'></div>
        <div style='position:absolute;left:0;top:0;height:100%;width:{p*100:.1f}%;background:{TIER_COLOR[risk_tier(p)]};opacity:.9'></div>
        <div style='position:absolute;left:{thr*100:.1f}%;top:0;height:100%;border-left:2px dashed #111'></div></div>
        <div style='display:flex;justify-content:space-between;font-size:11px;color:#555'><span>0%</span>
        <span>dashed line = decision threshold ({thr:.0%})</span><span>100%</span></div>""",
        unsafe_allow_html=True)


# ============================================================================= 1. RISK CALCULATOR
with tabs[0]:
    p = float(predict(BASE)[0])
    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.metric("Churn probability", f"{p:.1%}")
    c2.metric("Churn flag", "⚠️ Likely to churn" if p >= thr else "✅ Likely to stay")
    c3.markdown("**Risk tier**<br>" + tier_badge(p), unsafe_allow_html=True)
    gauge(p)
    st.caption("Risk tiers: Low < 30% · Medium 30–60% · High ≥ 60%. On the held-out test set the High tier churned "
               "≈84% of the time and the Low tier ≈9%.")

    st.subheader("Why this score?")
    contrib, Xt = contributions(BASE)
    row = contrib.iloc[0].drop("_bias")
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(8)
    cd = pd.DataFrame({"feature": [f"{k} = {Xt.iloc[0][k]:,.2f}" if abs(Xt.iloc[0][k]) < 1e4 else f"{k} = {Xt.iloc[0][k]:,.0f}" for k in top.index],
                       "impact": top.values})
    cd["direction"] = np.where(cd.impact > 0, "raises churn risk", "lowers churn risk")
    ch = alt.Chart(cd).mark_bar().encode(
        x=alt.X("impact:Q", title="Contribution to churn log-odds (SHAP)"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises churn risk", "lowers churn risk"], range=[RED, NAVY]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", alt.Tooltip("impact:Q", format=".2f")]).properties(height=280)
    st.altair_chart(ch, use_container_width=True)
    st.caption("Red bars push this customer towards churn; blue bars push towards staying. "
               "Values are exact TreeSHAP contributions from the XGBoost model.")

    with st.expander("Suggested retention actions for this profile"):
        acts = []
        if not active: acts.append("**Re-engage**: inactive members churn at ~27% vs ~14% for active ones. Trigger a personalised digital/app or relationship-manager touchpoint.")
        if nprod == 1: acts.append("**Deepen the relationship**: single-product customers churn at ~28% vs ~8% for two-product customers. Offer a relevant second product.")
        if nprod >= 3: acts.append("**Investigate product bundle**: customers with 3–4 products churn at 83–100% in this data. Do *not* cross-sell further; review pricing/service quality for multi-product holders.")
        if geo == "Germany": acts.append("**Germany market review**: churn is ~32% vs ~16% in France/Spain – check competitive pricing and service in this market.")
        if 45 <= age <= 62: acts.append("**Life-stage offer**: customers aged 45–62 churn at ~49% (vs ~20% overall). Consider retirement/wealth-planning propositions.")
        if balance >= 150000: acts.append("**Protect high balances**: high-balance customers carry the largest revenue at risk – assign a dedicated banker.")
        if not acts: acts.append("No specific red flags – maintain standard engagement.")
        for a in acts: st.markdown("- " + a)

# ============================================================================= 2. DISTRIBUTION
with tabs[1]:
    ts = test_scores()
    ts["tier"] = ts.p.map(risk_tier)
    st.subheader("Where does this customer sit in the portfolio?")
    st.caption("Distribution of predicted churn probabilities across the 2,000 held-out test customers (not seen during training).")
    bins = alt.Chart(ts).transform_bin("bin", "p", bin=alt.Bin(step=0.025)).transform_aggregate(
        count="count()", groupby=["bin", "bin_end"]).mark_bar(color=NAVY, opacity=.85).encode(
        x=alt.X("bin:Q", title="Predicted churn probability", scale=alt.Scale(domain=[0, 1])), x2="bin_end", y=alt.Y("count:Q", title="Customers"))
    rule = alt.Chart(pd.DataFrame({"x": [p], "l": ["Selected customer"]})).mark_rule(color=RED, size=3).encode(x="x:Q")
    trule = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color="black", strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart((bins + rule + trule).properties(height=320), use_container_width=True)
    pct = (ts.p < p).mean()
    st.info(f"The selected customer's score ({p:.1%}) is higher than **{pct:.0%}** of test-set customers. "
            "Red line = selected customer, dashed line = decision threshold.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Predicted probability by actual outcome**")
        ts["Outcome"] = ts.y.map({0: "Retained", 1: "Churned"})
        h = alt.Chart(ts).transform_bin("bin", "p", bin=alt.Bin(step=0.05)).transform_aggregate(
            count="count()", groupby=["bin", "bin_end", "Outcome"]).mark_bar(opacity=.75).encode(
            x=alt.X("bin:Q", title="Predicted probability"), x2="bin_end", y=alt.Y("count:Q", stack=None, title="Customers"),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Retained", "Churned"], range=[NAVY, RED])))
        st.altair_chart(h.properties(height=280), use_container_width=True)
    with c2:
        st.markdown("**Risk-tier composition (test set)**")
        tt = ts.groupby("tier").agg(customers=("y", "size"), actual_churn_rate=("y", "mean")).reindex(["High", "Medium", "Low"]).reset_index()
        tt["actual_churn_rate"] = (tt.actual_churn_rate * 100).round(1)
        bar = alt.Chart(tt).mark_bar().encode(x=alt.X("tier:N", sort=["High", "Medium", "Low"], title=None), y="customers:Q",
                                              color=alt.Color("tier:N", scale=alt.Scale(domain=list(TIER_COLOR), range=list(TIER_COLOR.values())), legend=None),
                                              tooltip=["tier", "customers", "actual_churn_rate"])
        st.altair_chart(bar.properties(height=280), use_container_width=True)
        st.dataframe(tt.rename(columns={"actual_churn_rate": "actual churn rate (%)"}), hide_index=True, use_container_width=True)

    st.subheader("Segment explorer")
    seg_col = st.selectbox("Break down average predicted risk by", ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard"])
    sg = ts.groupby(seg_col).agg(customers=("y", "size"), mean_predicted=("p", "mean"), actual_churn=("y", "mean")).reset_index()
    sgm = sg.melt(id_vars=[seg_col, "customers"], value_vars=["mean_predicted", "actual_churn"], var_name="measure", value_name="rate")
    st.altair_chart(alt.Chart(sgm).mark_bar().encode(x=alt.X(f"{seg_col}:N", title=seg_col), y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title="Churn rate"),
                    xOffset="measure:N", color=alt.Color("measure:N", scale=alt.Scale(range=[NAVY, RED])), tooltip=[seg_col, "customers", alt.Tooltip("rate:Q", format=".1%")]).properties(height=280),
                    use_container_width=True)
    st.caption("Predicted and actual rates track each other closely, which shows the model is well calibrated across segments.")

# ============================================================================= 3. FEATURE IMPORTANCE
with tabs[2]:
    st.subheader("What drives churn?")
    perm = pd.read_csv(ROOT / "outputs" / "permutation_importance.csv")
    shp = pd.read_csv(ROOT / "outputs" / "shap_importance.csv", index_col=0).iloc[:, 0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SHAP importance** (mean |impact| on log-odds; includes engineered features)")
        d = shp.reset_index(); d.columns = ["feature", "value"]
        st.altair_chart(alt.Chart(d).mark_bar(color=NAVY).encode(x="value:Q", y=alt.Y("feature:N", sort="-x", title=None), tooltip=["feature", alt.Tooltip("value:Q", format=".3f")]).properties(height=430), use_container_width=True)
    with c2:
        st.markdown("**Permutation importance** (drop in test ROC-AUC when shuffled)")
        st.altair_chart(alt.Chart(perm).mark_bar(color=RED).encode(x=alt.X("mean:Q", title="AUC drop"), y=alt.Y("feature:N", sort="-x", title=None), tooltip=["feature", alt.Tooltip("mean:Q", format=".4f")]).properties(height=430), use_container_width=True)
    st.info("**Reading the results:** age, number of products and activity status dominate. Geography (Germany) and gender add smaller effects; "
            "tenure, credit score, salary and credit-card ownership contribute almost nothing once the others are known.")

    st.subheader("Partial dependence (interactive)")
    feat = st.selectbox("Feature", ["Age", "NumOfProducts", "Balance", "CreditScore", "Tenure", "EstimatedSalary", "IsActiveMember"])
    grids = {"Age": np.arange(18, 81, 2), "NumOfProducts": np.array([1, 2, 3, 4]), "Balance": np.linspace(0, 220000, 25),
             "CreditScore": np.linspace(350, 850, 21), "Tenure": np.arange(0, 11), "EstimatedSalary": np.linspace(10000, 200000, 20), "IsActiveMember": np.array([0, 1])}

    @st.cache_data
    def pdp(feature):
        out = []
        for v in grids[feature]:
            Xg = X_te.copy(); Xg[feature] = v
            out.append(PIPE.predict_proba(Xg)[:, 1].mean())
        return pd.DataFrame({feature: grids[feature], "avg_churn_probability": out})

    pdf = pdp(feat)
    if feat in ("NumOfProducts", "IsActiveMember", "Tenure"):
        ch = alt.Chart(pdf).mark_bar(color=NAVY).encode(x=alt.X(f"{feat}:O"), y=alt.Y("avg_churn_probability:Q", axis=alt.Axis(format="%")))
    else:
        ch = alt.Chart(pdf).mark_line(color=NAVY, strokeWidth=3, point=True).encode(x=alt.X(f"{feat}:Q"), y=alt.Y("avg_churn_probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)))
    st.altair_chart(ch.properties(height=300), use_container_width=True)
    st.caption("Average predicted churn probability across the test set when the chosen feature is forced to each value, all else unchanged.")

    with st.expander("Logistic-regression benchmark: standardised effects on odds"):
        coef = pd.read_csv(ROOT / "outputs" / "logit_coefficients.csv", index_col=0).iloc[:, 0]
        cd = pd.DataFrame({"feature": coef.index, "odds_change": np.exp(coef.values) - 1})
        st.altair_chart(alt.Chart(cd).mark_bar().encode(x=alt.X("odds_change:Q", axis=alt.Axis(format="%"), title="Change in odds per +1 SD"),
                         y=alt.Y("feature:N", sort="-x", title=None), color=alt.condition(alt.datum.odds_change > 0, alt.value(RED), alt.value(NAVY))).properties(height=420), use_container_width=True)

# ============================================================================= 4. WHAT-IF
with tabs[3]:
    st.subheader("What-if scenario simulator")
    st.caption("The baseline is the sidebar profile. Adjust the levers below to see how churn probability moves.")
    bk = f"{active}{nprod}{balance}{age}{geo}{tenure}{credit}{salary}{card}"
    l1, l2, l3, l4 = st.columns(4)
    s_act = l1.toggle("Active member", value=bool(active), key=f"a{bk}")
    s_prod = l2.select_slider("Number of products", [1, 2, 3, 4], value=int(nprod), key=f"p{bk}")
    s_bal = l3.slider("Balance (€)", 0, 260000, int(balance), 5000, key=f"b{bk}")
    s_ten = l4.slider("Tenure (years)", 0, 10, int(tenure), key=f"t{bk}")
    l5, l6, l7 = st.columns(3)
    s_card = l5.toggle("Has credit card", value=bool(card), key=f"c{bk}")
    s_cs = l6.slider("Credit score", 300, 850, int(credit), key=f"cs{bk}")
    s_sal = l7.slider("Estimated salary (€)", 0, 200000, int(salary), 5000, key=f"s{bk}")
    SC = BASE.copy()
    SC.loc[0, ["IsActiveMember", "NumOfProducts", "Balance", "Tenure", "HasCrCard", "CreditScore", "EstimatedSalary"]] = [int(s_act), int(s_prod), float(s_bal), int(s_ten), int(s_card), int(s_cs), float(s_sal)]
    SC = SC.astype(BASE.dtypes.to_dict())
    p0, p1 = float(predict(BASE)[0]), float(predict(SC)[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline probability", f"{p0:.1%}")
    m2.metric("Scenario probability", f"{p1:.1%}", delta=f"{(p1-p0)*100:+.1f} pts", delta_color="inverse")
    m3.metric("Scenario flag", "⚠️ Likely to churn" if p1 >= thr else "✅ Likely to stay")
    cmp_df = pd.DataFrame({"Scenario": ["Baseline", "Scenario"], "p": [p0, p1]})
    st.altair_chart(alt.Chart(cmp_df).mark_bar(size=60).encode(x=alt.X("p:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Churn probability"),
                    y=alt.Y("Scenario:N", sort=None, title=None), color=alt.Color("Scenario:N", scale=alt.Scale(range=[GREY, NAVY]), legend=None)).properties(height=120), use_container_width=True)

    st.markdown("### Sensitivity sweeps for the baseline customer")
    sw1, sw2 = st.columns(2)
    with sw1:
        rows = []
        for a_ in (0, 1):
            for n_ in (1, 2, 3, 4):
                r = BASE.copy(); r.loc[0, "IsActiveMember"] = a_; r.loc[0, "NumOfProducts"] = n_
                rows.append(dict(Products=n_, Status="Active" if a_ else "Inactive", p=float(predict(r)[0])))
        st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(x="Products:O", y=alt.Y("p:Q", axis=alt.Axis(format="%"), title="Churn probability"), xOffset="Status:N",
                        color=alt.Color("Status:N", scale=alt.Scale(domain=["Inactive", "Active"], range=[RED, NAVY]))).properties(height=280, title="Engagement × product depth"), use_container_width=True)
    with sw2:
        ages = np.arange(18, 81, 2); rr = pd.concat([BASE] * len(ages), ignore_index=True); rr["Age"] = ages
        rr2 = rr.copy(); rr2["IsActiveMember"] = 1 - int(active)
        sd = pd.concat([pd.DataFrame({"Age": ages, "p": predict(rr), "Profile": "Current"}),
                        pd.DataFrame({"Age": ages, "p": predict(rr2), "Profile": "Active" if not active else "Inactive"})])
        st.altair_chart(alt.Chart(sd).mark_line(strokeWidth=3).encode(x="Age:Q", y=alt.Y("p:Q", axis=alt.Axis(format="%"), title="Churn probability"),
                        color=alt.Color("Profile:N", scale=alt.Scale(range=[NAVY, GREEN]))).properties(height=280, title="Risk across the age range"), use_container_width=True)
    if p1 > 0.5 and s_prod >= 3:
        st.warning("⚠️ Customers with 3–4 products churn at 83–100% in the historical data. The model reproduces this pattern; treat it as a "
                   "warning signal about the multi-product experience, not as evidence that adding products *causes* churn.")
    st.caption("What-if outputs are model associations learned from historical data, not guaranteed causal effects of an intervention.")

# ============================================================================= 5. PERFORMANCE
with tabs[4]:
    st.subheader("Model comparison (held-out test set, n = 2,000)")
    rows = []
    for n, r in METRICS["test"].items():
        rows.append({"Model": n, "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "Accuracy @0.5": r["@0.5"]["accuracy"],
                     "Precision @tuned": r["@tuned"]["precision"], "Recall @tuned": r["@tuned"]["recall"], "F1 @tuned": r["@tuned"]["f1"],
                     "Tuned threshold": r["threshold"], "CV ROC-AUC": METRICS["cv"][n]["roc_auc"][0]})
    mt = pd.DataFrame(rows).set_index("Model")
    st.dataframe(mt.style.format("{:.3f}").highlight_max(axis=0, color="#d5efe9"), use_container_width=True)
    st.caption("Thresholds were tuned on out-of-fold predictions from the training set only; the test set was used once for final reporting.")
    ts = test_scores()
    ytrue, pp = ts.y.values, ts.p.values
    pred = (pp >= thr).astype(int)
    tp, fp, fn, tn = ((pred == 1) & (ytrue == 1)).sum(), ((pred == 1) & (ytrue == 0)).sum(), ((pred == 0) & (ytrue == 1)).sum(), ((pred == 0) & (ytrue == 0)).sum()
    st.markdown(f"### Live metrics at threshold {thr:.2f}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", f"{(tp+tn)/len(ytrue):.1%}"); k2.metric("Precision", f"{tp/max(tp+fp,1):.1%}")
    k3.metric("Recall", f"{tp/max(tp+fn,1):.1%}"); k4.metric("F1", f"{2*tp/max(2*tp+fp+fn,1):.3f}"); k5.metric("Flagged customers", f"{tp+fp:,}")
    cm = pd.DataFrame({"Actual": ["Retained", "Retained", "Churned", "Churned"], "Predicted": ["Retain", "Churn", "Retain", "Churn"], "n": [tn, fp, fn, tp]})
    base_ch = alt.Chart(cm).encode(x=alt.X("Predicted:N", sort=["Retain", "Churn"]), y=alt.Y("Actual:N", sort=["Retained", "Churned"]))
    st.altair_chart((base_ch.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None)) +
                     base_ch.mark_text(fontSize=22, color="black").encode(text="n:Q")).properties(height=220, width=360, title="Confusion matrix"))
    st.markdown("### Precision–recall trade-off: move the threshold and watch the campaign size")
    ths = np.linspace(0.05, 0.9, 35)
    pr_rows = []
    for t in ths:
        pr_ = (pp >= t).astype(int); tp_ = ((pr_ == 1) & (ytrue == 1)).sum(); fp_ = ((pr_ == 1) & (ytrue == 0)).sum(); fn_ = ((pr_ == 0) & (ytrue == 1)).sum()
        pr_rows += [dict(t=t, metric="Precision", v=tp_ / max(tp_ + fp_, 1)), dict(t=t, metric="Recall", v=tp_ / max(tp_ + fn_, 1)), dict(t=t, metric="F1", v=2 * tp_ / max(2 * tp_ + fp_ + fn_, 1))]
    st.altair_chart(alt.Chart(pd.DataFrame(pr_rows)).mark_line(strokeWidth=3).encode(x=alt.X("t:Q", title="Threshold"), y=alt.Y("v:Q", title=None, axis=alt.Axis(format="%")),
                    color=alt.Color("metric:N", scale=alt.Scale(range=[NAVY, GREEN, RED]))).properties(height=280), use_container_width=True)
    dec = pd.read_csv(ROOT / "outputs" / "decile_table.csv")
    st.markdown("### Decile lift table")
    st.dataframe(dec.rename(columns={"decile": "Decile", "n": "Customers", "churners": "Churners", "avg_p": "Avg predicted", "rate": "Actual churn", "cum_capture": "Cumulative capture", "lift": "Lift"})
                 .style.format({"Avg predicted": "{:.1%}", "Actual churn": "{:.1%}", "Cumulative capture": "{:.1%}", "Lift": "{:.2f}x"}), hide_index=True, use_container_width=True)
    st.caption("Contacting only the riskiest 20% of customers captures ≈63% of all churners (3.1× better than random).")
    for f, cap in [("model_1_comparison.png", "ROC, precision-recall and metric comparison"), ("model_3_probability_calibration.png", "Probability separation and calibration")]:
        st.image(str(ROOT / "figures" / f), caption=cap)

# ============================================================================= 6. BATCH
with tabs[5]:
    st.subheader("Score a customer file")
    st.caption("Upload a CSV with the columns: " + ", ".join(RAW_INPUT_COLS) + ". Extra columns (e.g., CustomerId) are kept in the output.")
    up = st.file_uploader("CSV file", type="csv")
    use_demo = st.checkbox("Use the sample bank dataset instead", value=up is None)
    src = load_data() if (use_demo and up is None) else (pd.read_csv(up) if up else None)
    if src is not None:
        miss = [c for c in RAW_INPUT_COLS if c not in src.columns]
        if miss:
            st.error(f"Missing required columns: {miss}")
        else:
            out = src.copy(); out["churn_probability"] = predict(src); out["churn_flag"] = (out.churn_probability >= thr).astype(int); out["risk_tier"] = out.churn_probability.map(risk_tier)
            a, b, c = st.columns(3)
            a.metric("Customers scored", f"{len(out):,}"); b.metric("Flagged as likely churners", f"{out.churn_flag.sum():,}")
            c.metric("High-risk customers", f"{(out.risk_tier == 'High').sum():,}")
            if use_demo and up is None:
                st.caption("Note: the sample file includes the training rows, so scores on it are optimistic – use the test-set tabs for honest performance.")
            st.dataframe(out.sort_values("churn_probability", ascending=False).head(200), use_container_width=True)
            st.download_button("⬇️ Download scored file", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")
