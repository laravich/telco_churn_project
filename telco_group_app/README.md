# Telco Churn — Group Model Comparison

An interactive Streamlit app that compares predictions from different group members’ trained models for the same customer.

Enter customer details such as tenure, charges, contract and services. Select the models and compare their churn scores and predictions.

## Folder structure

- `app.py` — Customer form and prediction results.
- `model_support.py` — Feature preparation, model export and prediction helpers.
- `models/` — Saved model bundles (`.joblib` files).
- `requirements.txt` — Python dependencies.
- `data/` — Optional local training dataset; not required by the app.
- Training notebooks — Code used to train and export each member’s model.

The `.gitkeep` file keeps an empty `models/` folder in Git. It does not affect predictions.

## 1. Prepare your model

Train your selected model in your notebook.

Export the **complete fitted pipeline**, including preprocessing such as scaling, encoding and feature selection—not just a classifier trained on encoded arrays.

The app supplies these original customer columns:

```text
gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies,
Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges
```

`SeniorCitizen` is an integer: 0 or 1. Other categorical inputs use the standard Telco dataset values.

If your model requires additional inputs, extend the form or provide an input adapter.

## 2. Export your fitted model

Make `model_support.py` available in your notebook’s working directory.

For a fitted pipeline that accepts the original customer columns:

```python
from model_support import export_bundle

export_bundle(
    path="models/member_name.joblib",
    model=fitted_pipeline,
    member="Member name",
    threshold=selected_threshold,
    feature_mode="raw",
    positive_label=1
)
```

Replace:

- `member_name.joblib` with a unique filename.
- `fitted_pipeline` with your fitted pipeline variable.
- `Member name` with your name.
- `selected_threshold` with your model’s chosen threshold.
- `positive_label` with the class label representing churn. Use `"Yes"` if your target labels are `"Yes"` and `"No"`.

The model must support `predict_proba` and expose `classes_`.

The exported bundle contains the fitted pipeline, member name, threshold and input settings.

### Models with engineered features

Normally, use `feature_mode="raw"` and include feature engineering inside your pipeline.

Custom transformers must be defined in importable Python modules that are included with the app.

For the existing V2 model, use:

```python
export_bundle(
    path="models/lavanya_v2.joblib",
    model=final_model,
    member="Lavanya",
    threshold=final_threshold,
    feature_mode="lavanya_v2",
    positive_label=1
)
```

This feature mode adds:

- `InternetAddonCount`
- `AutomaticPayment`
- `NewMonthToMonth`

Use it only for models trained with those exact engineered inputs. The existing V2 threshold is **0.57**. Every other model should use its own chosen threshold.

Models from other frameworks may require an adapter to match the app’s prediction interface.

## 3. Add models to the app

Place each exported bundle inside `models/`, using distinct filenames:

```text
models/
  lavanya_v2.joblib
  member_2.joblib
  member_3.joblib
```

The app automatically discovers `.joblib` files.

A full merge of everyone’s training branches is not required. The app needs their exported model bundles, required dependencies and any custom transformer modules.

## 4. Install dependencies

Activate your Python environment, then run from the app folder:

```bash
python -m pip install -r requirements.txt
```

Use compatible package versions for training, export and app execution. Ideally, all members use the same environment.

Add any additional model libraries, such as XGBoost, LightGBM or CatBoost, to `requirements.txt`. Pin matching versions before deployment.

If only Streamlit is missing:

```bash
python -m pip install streamlit
```

## 5. Run the app

From the directory containing `app.py`:

```bash
python -m streamlit run app.py
```

Open the Local URL printed in the terminal.

1. Select the models to compare.
2. Enter the customer’s details.
3. Click **Compare predictions**.

Results include:

- Each model’s churn score.
- Each model’s saved threshold.
- Churn flag or predicted stay.
- A score comparison chart.
- Downloadable prediction results.

The app loads trained models without retraining them.

## 6. Share through Git

Commit the app code, README, dependencies, trusted model bundles and any required custom modules to the shared app branch.

Keep local datasets and generated outputs ignored unless the group intends to share them.

Example entries in the repository’s `.gitignore`:

```gitignore
telco_group_app/data/
__pycache__/
catboost_info/
outputs_v4/
```

Do not ignore the model bundles if they need to be included for deployment.

Only load trusted model files. Loading `.joblib` files can execute Python code.

## 7. Deploy on Streamlit

Choose the repository and shared app branch.

If the app is inside `telco_group_app/`, use this entry point:

```text
telco_group_app/app.py
```

Include model bundles, compatible dependencies and custom modules in the repository.

The training CSV is not needed for deployment.

## Understanding predictions

A churn score at or above a model’s saved threshold produces **Churn flagged**. Otherwise, it produces **Predicted stay**.

Scores are estimates, not guarantees. Scores from class-weighted models may not be calibrated probabilities.

Different models may use different thresholds, so similar scores can produce different decisions.

This app compares predictions, not model accuracy. To compare accuracy, evaluate all models on the same labeled holdout using agreed metrics.

Enter recorded total charges rather than calculating them from tenure and current monthly charges.

## Troubleshooting

| Problem | Check |
|---|---|
| No models appear | Put exported bundles in the `models/` folder beside `app.py`. |
| Missing bundle field | Export using `export_bundle`, rather than saving only the classifier. |
| Missing input column | Match the model’s input schema or extend the form. |
| Custom transformer cannot load | Include its importable Python module. |
| Package version warning | Match the versions used during training and export. |
| Streamlit is missing | Install it in the active Python environment. |
| Exported file is missing | Check the notebook’s working directory using `Path.cwd()`. |
| New model does not appear | Reload or restart the app. | 