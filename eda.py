import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from churn_utils import add_features
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150})
NAVY, RED, GREY, TEAL = "#1F3A5F", "#C0392B", "#9AA5B1", "#2A9D8F"
df = pd.read_csv("data/European_Bank.csv")
base = df.Exited.mean()

# 1. class balance + geography + gender
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].bar(["Retained", "Churned"], [(df.Exited == 0).sum(), (df.Exited == 1).sum()], color=[NAVY, RED])
for i, v in enumerate([(df.Exited == 0).sum(), (df.Exited == 1).sum()]):
    ax[0].text(i, v + 80, f"{v:,}\n({v/len(df):.1%})", ha="center")
ax[0].set_title("Class distribution"); ax[0].set_ylim(0, 9800)
for a, col in zip(ax[1:], ["Geography", "Gender"]):
    r = df.groupby(col).Exited.mean().sort_values(ascending=False)
    a.bar(r.index, r.values * 100, color=[RED if v > base else NAVY for v in r.values])
    a.axhline(base * 100, color=GREY, ls="--", lw=1); a.set_title(f"Churn rate by {col}"); a.set_ylabel("Churn rate (%)")
    for i, v in enumerate(r.values): a.text(i, v * 100 + 0.6, f"{v:.1%}", ha="center")
plt.tight_layout(); plt.savefig("figures/eda_1_overview.png"); plt.close()

# 2. engagement & product depth
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
r = df.groupby("NumOfProducts").Exited.agg(["mean", "count"])
ax[0].bar(r.index.astype(str), r["mean"] * 100, color=[NAVY, TEAL, RED, RED])
for i, (m, c) in enumerate(zip(r["mean"], r["count"])): ax[0].text(i, m * 100 + 1.5, f"{m:.1%}\n(n={c:,})", ha="center", fontsize=8)
ax[0].set_ylim(0, 120); ax[0].set_title("Churn by number of products"); ax[0].set_xlabel("Products held"); ax[0].set_ylabel("Churn rate (%)")
r = df.groupby("IsActiveMember").Exited.mean()
ax[1].bar(["Inactive", "Active"], r.values * 100, color=[RED, NAVY]); ax[1].set_title("Churn by activity status")
for i, v in enumerate(r.values): ax[1].text(i, v * 100 + 0.6, f"{v:.1%}", ha="center")
ax[1].set_ylabel("Churn rate (%)")
r = df.groupby("HasCrCard").Exited.mean()
ax[2].bar(["No card", "Has card"], r.values * 100, color=[GREY, GREY]); ax[2].set_title("Churn by credit-card ownership")
for i, v in enumerate(r.values): ax[2].text(i, v * 100 + 0.6, f"{v:.1%}", ha="center")
ax[2].set_ylabel("Churn rate (%)"); ax[2].set_ylim(0, 30)
plt.tight_layout(); plt.savefig("figures/eda_2_engagement.png"); plt.close()

# 3. age / balance / tenure
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
df["AgeBand"] = pd.cut(df.Age, [17, 29, 39, 49, 59, 100], labels=["18-29", "30-39", "40-49", "50-59", "60+"])
r = df.groupby("AgeBand", observed=True).Exited.mean()
ax[0].bar(r.index.astype(str), r.values * 100, color=[RED if v > 0.3 else NAVY for v in r.values])
for i, v in enumerate(r.values): ax[0].text(i, v * 100 + 1, f"{v:.0%}", ha="center")
ax[0].set_title("Churn by age band"); ax[0].set_ylabel("Churn rate (%)")
df["BalBand"] = pd.cut(df.Balance, [-1, 0, 75000, 125000, 1e9], labels=["Zero", "<75k", "75-125k", ">125k"])
r = df.groupby("BalBand", observed=True).Exited.mean()
ax[1].bar(r.index.astype(str), r.values * 100, color=NAVY); ax[1].set_title("Churn by balance band")
for i, v in enumerate(r.values): ax[1].text(i, v * 100 + 0.6, f"{v:.1%}", ha="center", fontsize=9)
ax[1].set_ylabel("Churn rate (%)")
r = df.groupby("Tenure").Exited.mean()
ax[2].plot(r.index, r.values * 100, marker="o", color=NAVY); ax[2].axhline(base * 100, color=GREY, ls="--", lw=1)
ax[2].set_title("Churn by tenure (years)"); ax[2].set_ylim(0, 30); ax[2].set_ylabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("figures/eda_3_age_balance_tenure.png"); plt.close()

# 4. distributions of key numerics by outcome
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, c in zip(ax, ["Age", "CreditScore", "Balance"]):
    sns.kdeplot(data=df, x=c, hue="Exited", fill=True, common_norm=False, palette=[NAVY, RED], ax=a, alpha=.35, legend=False)
    a.set_title(f"{c} distribution")
ax[0].legend(["Churned", "Retained"], frameon=False)
plt.tight_layout(); plt.savefig("figures/eda_4_distributions.png"); plt.close()

# 5. correlation heat map incl. engineered features
d2 = add_features(df.drop(columns=["Year", "CustomerId", "Surname"]))
d2 = pd.get_dummies(d2, columns=["Geography", "Gender"], drop_first=False).drop(columns=["AgeBand", "BalBand"])
corr = d2.astype(float).corr()
cols = corr["Exited"].abs().sort_values(ascending=False).index[:14]
plt.figure(figsize=(9, 7)); sns.heatmap(corr.loc[cols, cols], annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 7}, cbar_kws={"shrink": .7})
plt.title("Correlation matrix (top features by |corr| with churn)"); plt.tight_layout(); plt.savefig("figures/eda_5_correlation.png"); plt.close()
print(corr["Exited"].sort_values(ascending=False).round(3))
