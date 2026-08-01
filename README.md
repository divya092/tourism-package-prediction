# Tourism Package Prediction — MLOps Pipeline

Predicts whether a customer will purchase the Wellness Tourism Package for "Visit with Us",
with an end-to-end pipeline automated through GitHub Actions and deployed on Streamlit
Community Cloud.

## Repository structure

```
tourism_project/
├── data/tourism.csv                     registered dataset
├── model_building/
│   ├── data_register.py                 schema validation and summary
│   ├── prep.py                          cleaning and train/test split
│   └── train.py                         tuning, MLflow tracking, evaluation
├── deployment/
│   ├── app.py                           Streamlit application
│   ├── requirements.txt                 deployment dependencies
│   └── best_tourism_model_v1.joblib     committed by the pipeline
└── requirements.txt                     workflow dependencies
.github/workflows/pipeline.yml           CI/CD pipeline
```

## Pipeline

Triggered on every push to `main`:

1. **register-dataset** — validates the schema and prints a dataset summary.
2. **data-prep** — cleans, splits 80/20 stratified, uploads the splits as an artifact.
3. **model-traning** — downloads the splits, tunes XGBoost with `GridSearchCV`, logs every
   parameter set to MLflow, evaluates, and commits the trained model back to `main`.

## Model

XGBoost classifier inside a scikit-learn `Pipeline` (median/mode imputation, standard scaling,
one-hot encoding). Tuned on F1 over 96 parameter combinations with 5-fold cross-validation,
including `scale_pos_weight` to handle the ~19% positive class rate.

Approximate held-out test performance: F1 0.84, recall 0.81, ROC-AUC 0.96.

## Deployment

The Streamlit app loads the committed model and returns a purchase probability, with an
adjustable follow-up threshold so the marketing team can set its own precision/recall
trade-off.
