# Telco Churn Prediction — Group Project

A Constructor Nexademy group project exploring customer churn through data analysis, machine learning and an interactive Streamlit app.

Each group member independently analyzed the customer data, explored churn patterns, and experimented with predictive models. This branch brings our exported models together so users can compare their predictions for the same customer.

## Live App

[Open the Telco Churn Model Comparison App](https://telcochurnproject-rnryjxzqzappsudnguya92.streamlit.app/)

Enter a customer profile, select the available models, and see whether each model flags the customer as likely to churn or predicts that they will stay.

## Dataset

We use `telecom_users.csv`, containing 5,986 customer records.

Dataset reference: [Telecom Users Dataset — Kaggle, Radmir Zosimov](https://www.kaggle.com/datasets/radmirzosimov/telecom-users-dataset)

The dataset includes:

- Customer demographics.
- Tenure and contract type.
- Phone and internet services.
- Security, backup, protection, support and streaming subscriptions.
- Payment method and billing details.
- Monthly and total charges.
- Whether the customer churned.

The target column is `Churn`: `Yes` means the customer left, and `No` means the customer stayed.

## Individual Analysis and Modeling

Each member worked in a separate Git branch and performed their own analysis and modeling experiments.

Depending on the member’s approach, this included:

- Data cleaning and exploratory analysis.
- Investigating customer groups and churn rates.
- Feature engineering.
- Preprocessing numerical and categorical inputs.
- Comparing classification models.
- Evaluating predictions and selecting model settings.

Models may use different features, preprocessing methods and decision thresholds.

## What the App Does

The app uses one shared customer form and sends the same profile to each selected model.

Each model applies its own input preparation and fitted preprocessing. The app displays:

- Model and group member.
- Churn score.
- Saved decision threshold.
- Churn flag or predicted stay.
- A comparison chart.
- Downloadable prediction results.

The app loads already-trained models. It does not retrain them when a user changes customer details.

## Branch Structure

This branch is `merge_all_models_streamlit`.

The `telco_group_app/` folder contains:

| File or folder | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `model_support.py` | Shared model export and prediction helpers |
| `models/` | Exported model bundles |
| `Danial_features.py` | Input grouping used by Danial’s exported model |
| `requirements.txt` | App dependencies |
| `README.md` | Detailed setup and model export instructions |
| Notebooks | Training and export code |
| `data/` | Dataset used locally by the notebooks |

The training dataset is not required for app predictions once the models are exported.

## Run Locally

Use Python 3.11 and run from the repository root:

```bash
python -m pip install -r telco_group_app/requirements.txt
python -m streamlit run telco_group_app/app.py
```

Open the Local URL printed in the terminal.

The deployed environment uses `scikit-learn==1.4.2` to match the existing exported models.

## Add a Group Member’s Model

1. Train and select the model in the member’s notebook.
2. Include all fitted preprocessing and required feature preparation.
3. Export a bundle using `export_bundle()` from `model_support.py`.
4. Place it in `telco_group_app/models/` with a unique filename.
5. Include any custom transformer modules and required dependencies.
6. Test predictions locally, then commit and push to this branch.

See [the app README](telco_group_app/README.md) for detailed export instructions.

## Interpreting the Comparison

A churn score at or above a model’s saved threshold produces a churn flag. Otherwise, the model predicts stay.

These are estimates, not guarantees. Class-weighted models may produce scores that are not calibrated probabilities.

Different models can disagree because they learned different relationships or use different thresholds.

The app compares individual predictions; it does not establish which model is most accurate. A fair performance comparison requires the same labeled evaluation data and agreed metrics.

## Acknowledgments

Developed collaboratively as part of the Constructor Nexademy Data Science training program.

Dataset credit: Radmir Zosimov’s Telecom Users Dataset on Kaggle.