import json, warnings, joblib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, shap
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from churn_utils import risk_tier
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150})
NAVY, RED, GREY, TEAL = "#1F3A5F", "#C0392B", "#9AA5B1", "#2A9D8F"
art = joblib.load("models/churn_model.joblib"); X_tr, X_te, y_tr, y_te = joblib.load("models/split.joblib")
metrics = json.load(open("outputs/metrics.json"))
champ, pipe, thr = art["name"], art["pipeline"], art["threshold"]
allp, allthr = art["all_pipelines"], art["thresholds"]
proba = pipe.predict_proba(X_te)[:, 1]; facts = {"champion": champ, "threshold": thr}

# ---- model comparison (ROC / PR / metric bars)
colors = dict(zip(allp, [GREY, "#E9A23B", TEAL, NAVY, RED]))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for n, p in allp.items():
    pp = p.predict_proba(X_te)[:, 1]; f, t, _ = roc_curve(y_te, pp)
    ax[0].plot(f, t, color=colors[n], lw=2.5 if n == champ else 1.4, label=f"{n} ({metrics['test'][n]['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(y_te, pp); ax[1].plot(rc, pr, color=colors[n], lw=2.5 if n == champ else 1.4, label=n)
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_title("ROC curves (test set)"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(frameon=False, fontsize=8, loc="lower right")
ax[1].axhline(y_te.mean(), color="k", ls=":", lw=1); ax[1].set_title("Precision-recall curves (test set)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
names = list(allp); w = 0.2
for i, m in enumerate(["precision", "recall", "f1"]):
    ax[2].bar(np.arange(len(names)) + (i - 1) * 0.27, [metrics["test"][n]["@tuned"][m] for n in names], 0.27, label=m.capitalize(), color=[NAVY, TEAL, RED][i])
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8); ax[2].set_ylim(0, 0.85); ax[2].legend(frameon=False, fontsize=8, ncol=3, loc="upper center"); ax[2].set_title("Precision / recall / F1 at tuned threshold")
plt.tight_layout(); plt.savefig("figures/model_1_comparison.png"); plt.close()

# ---- confusion matrix at tuned threshold and at 0.5
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, t, ttl in zip(ax, [0.5, thr], ["Threshold 0.50", f"Tuned threshold {thr:.2f}"]):
    cm = confusion_matrix(y_te, (proba >= t).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=["Pred retain", "Pred churn"], yticklabels=["Actual retain", "Actual churn"])
    a.set_title(f"{champ}: {ttl}")
plt.tight_layout(); plt.savefig("figures/model_2_confusion.png"); plt.close()
tn, fp, fn, tp = confusion_matrix(y_te, (proba >= thr).astype(int)).ravel()
facts["cm_tuned"] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

# ---- probability distribution + calibration
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(proba[y_te == 0], bins=30, alpha=.65, color=NAVY, label="Retained", density=True)
ax[0].hist(proba[y_te == 1], bins=30, alpha=.65, color=RED, label="Churned", density=True)
ax[0].axvline(thr, color="k", ls="--", lw=1); ax[0].text(thr + .01, ax[0].get_ylim()[1] * .9, f"threshold {thr:.2f}", fontsize=8)
ax[0].set_title("Predicted churn probability by actual outcome"); ax[0].set_xlabel("Predicted probability"); ax[0].legend(frameon=False)
fr, mp = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")
ax[1].plot(mp, fr, marker="o", color=NAVY); ax[1].plot([0, 1], [0, 1], "k:", lw=1); ax[1].set_title("Calibration (reliability) curve")
ax[1].set_xlabel("Mean predicted probability"); ax[1].set_ylabel("Observed churn rate")
plt.tight_layout(); plt.savefig("figures/model_3_probability_calibration.png"); plt.close()

# ---- gains / lift / risk tiers
d = pd.DataFrame({"p": proba, "y": y_te.values}).sort_values("p", ascending=False).reset_index(drop=True)
d["decile"] = pd.qcut(d.index, 10, labels=range(1, 11))
dec = d.groupby("decile", observed=True).agg(n=("y", "size"), churners=("y", "sum"), avg_p=("p", "mean"))
dec["rate"] = dec.churners / dec.n; dec["cum_capture"] = dec.churners.cumsum() / dec.churners.sum(); dec["lift"] = dec.rate / d.y.mean()
dec.to_csv("outputs/decile_table.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(dec.index.astype(int), dec.lift, color=[RED if l > 1.5 else NAVY for l in dec.lift]); ax[0].axhline(1, color="k", ls=":", lw=1)
ax[0].set_title("Lift by risk decile"); ax[0].set_xlabel("Decile (1 = highest predicted risk)"); ax[0].set_ylabel("Lift vs. base rate")
ax[1].plot([0] + list(dec.index.astype(int) * 10), [0] + list(dec.cum_capture * 100), marker="o", color=NAVY, label="Model")
ax[1].plot([0, 100], [0, 100], "k:", label="Random"); ax[1].set_title("Cumulative gains"); ax[1].set_xlabel("% of customers contacted (highest risk first)"); ax[1].set_ylabel("% of churners captured"); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig("figures/model_4_lift_gains.png"); plt.close()
facts["decile"] = dec.round(4).reset_index().to_dict("records")
tiers = pd.Series([risk_tier(p) for p in proba]); tt = pd.DataFrame({"tier": tiers, "y": y_te.values})
tier_tbl = tt.groupby("tier").agg(customers=("y", "size"), churners=("y", "sum")); tier_tbl["actual_rate"] = tier_tbl.churners / tier_tbl.customers
tier_tbl["share_of_all_churners"] = tier_tbl.churners / tier_tbl.churners.sum(); tier_tbl = tier_tbl.reindex(["High", "Medium", "Low"])
tier_tbl.to_csv("outputs/risk_tiers.csv"); facts["tiers"] = tier_tbl.round(4).reset_index().to_dict("records"); print(tier_tbl)

# ---- permutation importance (raw inputs, AUC drop)
pi = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=15, random_state=42, n_jobs=1)
pim = pd.DataFrame({"feature": X_te.columns, "mean": pi.importances_mean, "std": pi.importances_std}).sort_values("mean")
fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(pim.feature, pim["mean"], xerr=pim["std"], color=NAVY); ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
ax.set_title(f"Permutation importance ({champ})"); plt.tight_layout(); plt.savefig("figures/xai_1_permutation.png"); plt.close()
pim.sort_values("mean", ascending=False).to_csv("outputs/permutation_importance.csv", index=False); facts["perm"] = pim.sort_values("mean", ascending=False).round(4).to_dict("records")

# ---- SHAP on engineered/encoded features
Xt = pipe[:-1].transform(X_te); model = pipe[-1]
expl = shap.TreeExplainer(model); sv = expl.shap_values(Xt)
sv = sv[1] if isinstance(sv, list) else sv
plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=15); plt.title("SHAP summary (impact on log-odds of churn)"); plt.tight_layout(); plt.savefig("figures/xai_2_shap_beeswarm.png", bbox_inches="tight"); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(imp.index, imp.values, color=NAVY); ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Global SHAP feature importance"); plt.tight_layout(); plt.savefig("figures/xai_3_shap_bar.png"); plt.close()
imp.sort_values(ascending=False).to_csv("outputs/shap_importance.csv", header=["mean_abs_shap"]); facts["shap"] = imp.sort_values(ascending=False).round(4).to_dict()
print(imp.sort_values(ascending=False).round(3))
# SHAP dependence for the top interactions
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, f in zip(ax, ["Age", "NumOfProducts", "Balance"]):
    shap.dependence_plot(f, sv, Xt, ax=a, show=False, interaction_index=None); a.set_title(f"SHAP dependence: {f}")
plt.tight_layout(); plt.savefig("figures/xai_4_shap_dependence.png"); plt.close()

# ---- Partial dependence (manual, robust to integer columns)
def pdp(feature, grid):
    out = []
    for v in grid:
        Xg = X_te.copy(); Xg[feature] = v; out.append(pipe.predict_proba(Xg)[:, 1].mean())
    return np.array(out)
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5)); pdp_out = {}
specs = {"Age": np.arange(20, 81, 3), "NumOfProducts": np.array([1, 2, 3, 4]), "Balance": np.linspace(0, 220000, 30),
         "CreditScore": np.linspace(350, 850, 26), "Tenure": np.arange(0, 11), "IsActiveMember": np.array([0, 1])}
for a_, (f, g) in zip(ax.ravel(), specs.items()):
    v = pdp(f, g); pdp_out[f] = dict(zip([float(x) for x in g], [float(x) for x in v]))
    if f in ("NumOfProducts", "IsActiveMember"): a_.bar([str(int(x)) for x in g], v * 100, color=NAVY)
    else: a_.plot(g, v * 100, color=NAVY, lw=2)
    a_.set_title(f); a_.set_ylabel("Avg. predicted churn (%)")
plt.suptitle("Partial dependence plots", fontweight="bold"); plt.tight_layout(); plt.savefig("figures/xai_5_pdp.png"); plt.close()
facts["pdp"] = pdp_out

# ---- Logistic regression odds ratios (interpretability benchmark)
lr = allp["Logistic Regression"]; names_lr = lr[:-1].transform(X_te.head(2)).columns
coef = pd.Series(lr[-1].coef_[0], index=names_lr).sort_values()
fig, ax = plt.subplots(figsize=(7, 5.5)); ax.barh(coef.index, np.exp(coef.values) - 1, color=[RED if c > 0 else NAVY for c in coef.values])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Change in odds of churn per +1 SD (numeric) / vs. absent (categorical)"); ax.set_title("Logistic regression: standardised effects on odds")
plt.tight_layout(); plt.savefig("figures/xai_6_logit_odds.png"); plt.close(); coef.to_csv("outputs/logit_coefficients.csv", header=["coef"])

# ---- segment view on test set: actual vs predicted
seg = X_te.copy(); seg["y"] = y_te.values; seg["p"] = proba
seg["AgeBand"] = pd.cut(seg.Age, [17, 39, 49, 59, 100], labels=["<40", "40-49", "50-59", "60+"])
rows = []
for col in ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "AgeBand"]:
    g = seg.groupby(col, observed=True).agg(n=("y", "size"), actual=("y", "mean"), predicted=("p", "mean")).reset_index().rename(columns={col: "value"}); g.insert(0, "segment", col); rows.append(g)
pd.concat(rows).round(4).to_csv("outputs/segment_calibration.csv", index=False)

# ---- What-if scenario illustrations (used in paper)
base = pd.DataFrame([dict(CreditScore=650, Geography="Germany", Gender="Female", Age=45, Tenure=5, Balance=120000.0, NumOfProducts=1, HasCrCard=1, IsActiveMember=0, EstimatedSalary=100000.0)])
sc = {"Baseline (inactive, 1 product)": base}
b2 = base.copy(); b2["IsActiveMember"] = 1; sc["Re-engaged (active)"] = b2
b3 = base.copy(); b3["NumOfProducts"] = 2; sc["Cross-sold to 2 products"] = b3
b4 = b2.copy(); b4["NumOfProducts"] = 2; sc["Active + 2 products"] = b4
b5 = b4.copy(); b5["NumOfProducts"] = 3; sc["Active + 3 products"] = b5
scen = {k: float(pipe.predict_proba(v)[0, 1]) for k, v in sc.items()}; print(scen); facts["scenario"] = scen
fig, ax = plt.subplots(figsize=(8, 3.5)); ax.barh(list(scen)[::-1], [v * 100 for v in list(scen.values())[::-1]], color=[NAVY, TEAL, TEAL, TEAL, RED][::-1] if False else NAVY)
for i, v in enumerate(list(scen.values())[::-1]): ax.text(v * 100 + 1, i, f"{v:.0%}", va="center")
ax.set_xlabel("Predicted churn probability (%)"); ax.set_xlim(0, 100); ax.set_title("What-if: 45-year-old German customer, EUR 120k balance"); plt.tight_layout(); plt.savefig("figures/scenario_whatif.png"); plt.close()
json.dump(facts, open("outputs/facts.json", "w"), indent=2, default=float)
print("done")
