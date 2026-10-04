"""Keep this module next to app.py and the export notebook for joblib loading."""
def prepare_grouped_inputs(X):
    X = X.copy()
    X['contract2'] = X['Contract'].replace({'One year': 'Yearly', 'Two year': 'Yearly'})
    X = X.drop(columns='Contract')
    X['PaymentMethod2'] = X['PaymentMethod'].replace({
        'Mailed check': 'Necessary',
        'Bank transfer (automatic)': 'Necessary',
        'Credit card (automatic)': 'Necessary'
    })
    return X.drop(columns='PaymentMethod')
