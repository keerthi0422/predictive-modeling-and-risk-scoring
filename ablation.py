import json, joblib, warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import churn_utils as cu
warnings.filterwarnings("ignore")
X_tr, X_te, y_tr, y_te = joblib.load("models/split.joblib")
m = json.load(open("outputs/metrics.json")); bp = m["test"]["XGBoost"]["best_params"]
def build(num, cat):
    prep = ColumnTransformer([("num", "passthrough", num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)], verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("f", FunctionTransformer(cu.add_features, validate=False)), ("p", prep),
                     ("m", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1, **bp))])
raw, eng = cu.RAW_NUMERIC, cu.ENGINEERED
sets = {
 "All features (champion)": (raw + eng, ["Geography", "Gender"]),
 "Raw features only (no engineering)": (raw, ["Geography", "Gender"]),
 "Without Gender": ([c for c in raw + eng], ["Geography"]),
 "Without Age-based features (Age, AgeTenure)": ([c for c in raw + eng if c not in ("Age", "AgeTenure")], ["Geography", "Gender"]),
 "Behavioural only (products, activity, balance, tenure, card, engineered)": (["Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "BalanceSalaryRatio", "ProductDensity", "EngagementProduct", "ZeroBalance"], []),
 "Demographics only (Age, Gender, Geography)": (["Age"], ["Geography", "Gender"]),
}
skf = StratifiedKFold(5, shuffle=True, random_state=42); rows = []
for k, (num, cat) in sets.items():
    p = build(num, cat)
    cv = cross_val_score(p, X_tr, y_tr, cv=skf, scoring="roc_auc").mean()
    p.fit(X_tr, y_tr); te = roc_auc_score(y_te, p.predict_proba(X_te)[:, 1])
    rows.append(dict(setup=k, cv_auc=cv, test_auc=te)); print(f"{k:75s} CV {cv:.4f} test {te:.4f}")
pd.DataFrame(rows).to_csv("outputs/ablation.csv", index=False)
