"""Train, tune, evaluate and persist churn models."""
import json, warnings, joblib
import numpy as np, pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import (train_test_split, StratifiedKFold, RandomizedSearchCV,
                                     cross_val_predict, cross_validate)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, precision_recall_curve)
from xgboost import XGBClassifier
from churn_utils import make_pipeline, RAW_INPUT_COLS

warnings.filterwarnings("ignore")
SEED = 42
df = pd.read_csv("data/European_Bank.csv")
X, y = df[RAW_INPUT_COLS], df["Exited"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
print("train", X_tr.shape, y_tr.mean().round(4), "| test", X_te.shape, y_te.mean().round(4))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

candidates = {
    "Logistic Regression": (make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0), scale=True), None),
    "Decision Tree": (make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=SEED), scale=False),
                      {"model__max_depth": randint(3, 9), "model__min_samples_leaf": randint(20, 120),
                       "model__min_samples_split": randint(20, 200)}),
    "Random Forest": (make_pipeline(RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                                           n_jobs=1, random_state=SEED), scale=False),
                      {"model__max_depth": randint(4, 14), "model__min_samples_leaf": randint(5, 40),
                       "model__max_features": ["sqrt", 0.4, 0.6]}),
    "Gradient Boosting": (make_pipeline(GradientBoostingClassifier(random_state=SEED), scale=False),
                          {"model__n_estimators": randint(100, 300), "model__learning_rate": uniform(0.02, 0.13),
                           "model__max_depth": randint(2, 5), "model__subsample": uniform(0.6, 0.4),
                           "model__min_samples_leaf": randint(10, 60)}),
    "XGBoost": (make_pipeline(XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=SEED,
                                            scale_pos_weight=1.0), scale=False),
                {"model__n_estimators": randint(100, 350), "model__learning_rate": uniform(0.02, 0.13),
                 "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4),
                 "model__colsample_bytree": uniform(0.6, 0.4), "model__min_child_weight": randint(1, 10),
                 "model__reg_lambda": uniform(0.5, 4)}),
}

def best_f1_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = np.nanargmax(f1[:-1])
    return float(th[i])

fitted, results, thresholds, cv_summary = {}, {}, {}, {}
for name, (pipe, space) in candidates.items():
    if space:
        search = RandomizedSearchCV(pipe, space, n_iter=8, scoring="roc_auc", cv=skf,
                                    random_state=SEED, n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        est = search.best_estimator_
        best_params = {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                       for k, v in search.best_params_.items()}
    else:
        est = pipe.fit(X_tr, y_tr); best_params = {"C": 1.0, "class_weight": "balanced"}
    # 5-fold CV metrics on training set at default 0.5 threshold
    cv = cross_validate(est, X_tr, y_tr, cv=skf, n_jobs=1,
                        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    cv_summary[name] = {m: (float(cv[f"test_{m}"].mean()), float(cv[f"test_{m}"].std()))
                        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    # Out-of-fold probabilities -> choose F1-optimal threshold WITHOUT touching the test set
    oof = cross_val_predict(est, X_tr, y_tr, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    thr = best_f1_threshold(y_tr, oof)
    p_te = est.predict_proba(X_te)[:, 1]
    row = {"threshold": thr, "best_params": best_params}
    for label, t in [("@0.5", 0.5), ("@tuned", thr)]:
        pred = (p_te >= t).astype(int)
        row[label] = dict(accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred),
                          recall=recall_score(y_te, pred), f1=f1_score(y_te, pred))
    row["roc_auc"] = roc_auc_score(y_te, p_te); row["pr_auc"] = average_precision_score(y_te, p_te)
    results[name] = row; fitted[name] = est; thresholds[name] = thr
    print(f"{name:20s} CV-AUC {cv_summary[name]['roc_auc'][0]:.4f} | test AUC {row['roc_auc']:.4f} "
          f"| @0.5 P/R/F1 {row['@0.5']['precision']:.3f}/{row['@0.5']['recall']:.3f}/{row['@0.5']['f1']:.3f} "
          f"| tuned thr {thr:.2f} P/R/F1 {row['@tuned']['precision']:.3f}/{row['@tuned']['recall']:.3f}/{row['@tuned']['f1']:.3f}")

# Select champion on cross-validated ROC-AUC (test set untouched for selection)
champion = max(cv_summary, key=lambda k: cv_summary[k]["roc_auc"][0])
print("CHAMPION:", champion)
joblib.dump({"name": champion, "pipeline": fitted[champion], "threshold": thresholds[champion],
             "all_pipelines": fitted, "thresholds": thresholds}, "models/churn_model.joblib")
joblib.dump((X_tr, X_te, y_tr, y_te), "models/split.joblib")
json.dump({"champion": champion, "cv": cv_summary, "test": results}, open("outputs/metrics.json", "w"), indent=2)
