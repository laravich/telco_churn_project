
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.inspection import permutation_importance

# 1. Load the dataset
df = pd.read_csv("telecom_users.csv")

# Clean and prepare the data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Remove the customer ID from predictive features
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# 2. Explore churn distribution
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn distribution:\n", y.value_counts(normalize=True))

# 3. Analyze churn rates for selected features
features = [
    "Contract", "InternetService", "TechSupport",
    "PaymentMethod", "OnlineSecurity"
]

for feature in features:
    churn_rates = df.groupby(feature, observed=True)["Churn"].mean()
    print(f"\nChurn rate by {feature}:\n")
    print((churn_rates * 100).round(2))

# 4. Prepare numerical and categorical features
numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 7. Evaluate the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Churn Prediction ROC Curve")
plt.show()

# 8. Identify the most influential features
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)

importance = pd.Series(
    perm.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)

importance.sort_values().plot(
    kind="barh", figsize=(9, 6)
)
plt.title("Permutation Feature Importance")
plt.xlabel("Mean decrease in ROC-AUC")
plt.tight_layout()
plt.show()

# 9. Identify customers with high predicted churn risk
results = X_test.copy()
results["ActualChurn"] = y_test
results["ChurnProbability"] = y_prob
results["PredictedChurn"] = y_pred

results = results.sort_values(
    "ChurnProbability", ascending=False
)

print("\nTop 20 customers by predicted churn risk:")
print(results.head(20))

# Save predictions
results.to_csv("churn_predictions.csv", index=True)