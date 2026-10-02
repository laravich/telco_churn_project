from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from model_support import predict_bundle

st.set_page_config(page_title='Telco · Model comparison', page_icon='📡', layout='wide')
st.title('📡 Telco customer predictions')
st.write('Enter one customer profile and compare predictions from your group’s trained models.')
st.caption('A churn flag uses each model’s own threshold. Scores are model estimates, not guarantees; class-weighted scores may not be calibrated probabilities.')

model_dir = Path(__file__).parent / 'models'
paths = sorted(model_dir.glob('*.joblib'))
if not paths:
    st.info('No trained models installed yet. Follow README.md to export model bundles into models/. You can explore the form now.')

@st.cache_resource
def load_bundle(path, modified):
    return joblib.load(path)

bundles = []
for path in paths:
    try:
        b = load_bundle(str(path), path.stat().st_mtime_ns)
        for key in ['model', 'member', 'threshold', 'feature_mode', 'positive_label']:
            if key not in b:
                raise ValueError(f'Missing bundle field: {key}')
        bundles.append((path.stem, b))
    except Exception as exc:
        st.error(f'Could not load {path.name}: {exc}')
selected = st.multiselect('Models to compare', [key for key, b in bundles], default=[key for key, b in bundles])

# These choices reproduce the standard Telco raw schema.
a, b, c = st.columns(3)
with a:
    st.subheader('Customer')
    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior = st.selectbox('Senior citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12, step=1)
with b:
    st.subheader('Account and billing')
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment = st.selectbox('Payment method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless = st.selectbox('Paperless billing', ['Yes', 'No'])
    monthly = st.number_input('Current monthly charges (dataset currency)', min_value=0.0, value=70.0, step=1.0)
    total = st.number_input('Recorded total charges (dataset currency)', min_value=0.0, value=840.0, step=10.0)
    st.caption('Use the recorded total. It need not equal tenure × current monthly charges.')
with c:
    st.subheader('Services')
    phone = st.selectbox('Phone service', ['Yes', 'No'])
    lines = st.selectbox('Multiple lines', ['No', 'Yes']) if phone == 'Yes' else 'No phone service'
    internet = st.selectbox('Internet service', ['DSL', 'Fiber optic', 'No'])
    addons = {}
    for column, label in [('OnlineSecurity', 'Online security'), ('OnlineBackup', 'Online backup'), ('DeviceProtection', 'Device protection'), ('TechSupport', 'Technical support'), ('StreamingTV', 'Streaming TV'), ('StreamingMovies', 'Streaming movies')]:
        addons[column] = st.selectbox(label, ['No', 'Yes'], key=column) if internet != 'No' else 'No internet service'

raw = pd.DataFrame([dict(gender=gender, SeniorCitizen=int(senior == 'Yes'), Partner=partner,
    Dependents=dependents, tenure=int(tenure), PhoneService=phone, MultipleLines=lines,
    InternetService=internet, **addons, Contract=contract, PaperlessBilling=paperless,
    PaymentMethod=payment, MonthlyCharges=float(monthly), TotalCharges=float(total))])
if st.button('Compare predictions', type='primary'):
    if not selected:
        st.warning('Install and select at least one trained model.')
    rows = []
    for key, bundle in bundles:
        if key not in selected:
            continue
        try:
            rows.append(dict(Model=key, **predict_bundle(bundle, raw)))
        except Exception as exc:
            st.error(f'{key}: {exc}')
    if rows:
        results = pd.DataFrame(rows)
        st.subheader('Same customer · each model’s decision')
        st.dataframe(results.style.format({'Churn_score': '{:.1%}', 'Threshold': '{:.2f}'}), hide_index=True, use_container_width=True)
        st.bar_chart(results.set_index('Model')[['Churn_score']], horizontal=True)
        st.caption('Different thresholds can produce different labels for similar scores. Agreement does not establish accuracy; compare performance using the same labeled holdout.')
        st.download_button('Download predictions', results.to_csv(index=False), 'customer_predictions.csv', 'text/csv')
with st.expander('Customer inputs sent to the models'):
    st.dataframe(raw, hide_index=True)
