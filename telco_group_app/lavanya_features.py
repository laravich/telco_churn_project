"""Feature preparation for Lavanya's V2 model."""

def prepare_lavanya_inputs(X):
    X = X.copy()

    addons = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    X["InternetAddonCount"] = X[addons].eq("Yes").sum(axis=1)

    X["AutomaticPayment"] = X["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

    X["NewMonthToMonth"] = (
        X["tenure"].le(6)
        & X["Contract"].eq("Month-to-month")
    ).astype(int)

    return X