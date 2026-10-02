# Telco group model comparison

One shared customer form, multiple fitted models, individual thresholds, comparison table, score chart and CSV download. No retraining happens in the app. No trained model is included: a notebook contains code and outputs, not its live fitted Python objects.

## 1. Export Lavanya’s exact V2 model

Run V2 through the cell that fits `final_model`. Put `model_support.py` next to the notebook (or add the app directory to Python's import path), then run:

```python
from model_support import export_bundle
export_bundle('models/lavanya_v2.joblib', final_model,
              member='Lavanya', threshold=final_threshold,
              feature_mode='lavanya_v2', positive_label=1)
```

V2 sets `final_threshold = 0.57`. It uses balanced logistic regression with StandardScaler, OneHotEncoder and three externally engineered features. This app reproduces InternetAddonCount, AutomaticPayment and NewMonthToMonth exactly and retains the original service columns. Copy the resulting bundle into this app's models/ directory.

## 2. Add group members

Each member must export their FITTED complete preprocessing + classifier pipeline, accepting the original Telco DataFrame columns:

```python
from model_support import export_bundle
export_bundle('models/member_name.joblib', fitted_pipeline,
              member='Member name', threshold=0.5,
              feature_mode='raw', positive_label=1)
```

Replace 0.5 with that member's actual frozen threshold. If Churn was trained as Yes/No strings, use positive_label='Yes'. Do not pass only an estimator trained on encoded arrays; package its fitted preprocessing too. Custom feature engineering should live in that member’s pipeline, with custom classes/functions in importable modules shipped with the app. PyCaret pipelines can be exported if they accept raw DataFrames and provide classes_ and predict_proba; other wrappers may need an adapter. Extra input columns require extending the form. Multiple models per member are supported with separate filenames.

Export with the same Python/scikit-learn/pandas/numpy and other model-library versions as the app environment. Prefer that all group members use one shared environment; replace requirements.txt version ranges with the actual matching versions. Add xgboost/lightgbm/catboost etc. if required by your saved pipelines. Only put trusted group model files in models/: joblib uses pickle and executes code when loading. The public app intentionally has no model-file uploader.

## 3. Run

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, commit app.py, model_support.py, requirements.txt, required custom modules and trusted models/*.joblib to your repository. Set app.py as the entry point. No CSV or training data is required for inference.

Scores from balanced classifiers need not be calibrated probabilities. The UI labels a score over the saved cutoff as “Churn flagged”; a 0.57 cutoff does not mean every flagged customer has over 50% true churn probability. App comparisons are demonstrations, not an accuracy ranking. Evaluate group models on a common holdout to compare performance.
