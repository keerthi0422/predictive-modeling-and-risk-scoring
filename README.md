# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Quick start
```
pip install -r requirements.txt
streamlit run app.py
```
The trained model is included (`models/churn_model.joblib`), so the dashboard runs immediately.
Note: the saved model was built with scikit-learn 1.8.0 / XGBoost 3.x – if you use different versions, re-run `python train.py` first.

## Reproduce the analysis (random seed 42)
```
python train.py      # split, tune, evaluate 5 models, choose champion + threshold  (~8 min on 1 CPU)
python explain.py    # SHAP, permutation importance, PDP, calibration, lift, risk tiers, what-if
python eda.py        # exploratory figures
python ablation.py   # feature-set ablation
```

## Contents
| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (risk calculator, probability distribution, feature importance, what-if simulator, model performance, batch scoring) |
| `churn_utils.py` | Feature engineering + preprocessing pipeline shared by all scripts |
| `data/European_Bank.csv` | Source data |
| `models/` | Trained pipelines (champion: XGBoost) and the train/test split |
| `figures/`, `outputs/` | All figures and metric tables used in the paper |
| `outputs/Churn_Research_Paper.docx` | Research paper (EDA, modelling, explainability, recommendations) |
| `outputs/Churn_Executive_Summary.docx` | 2-page executive summary |

## Headline results (held-out test set, n = 2,000)
XGBoost ROC-AUC 0.870 · precision 62% / recall 65% at tuned threshold 0.31 · top 20% of customers capture 63% of churners.
