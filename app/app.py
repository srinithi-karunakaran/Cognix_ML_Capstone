"""Telco Customer Churn predictor - Streamlit GUI for the 23CSE301 capstone (classification track).

Run from the repository root:   streamlit run app/app.py
"""
import pandas as pd
import streamlit as st

import inference

st.set_page_config(page_title='Telco Churn Predictor', page_icon='📞', layout='wide')


@st.cache_resource(show_spinner='Loading the trained model...')
def get_artifacts():
    return inference.load_artifacts()


try:
    model, prep = get_artifacts()
except FileNotFoundError:
    st.error('Model files not found in `models/`. Run `notebooks/classification.ipynb` top to bottom first '
             '(its last cells save `best_classifier.pkl` and `preprocessing.pkl`).')
    st.stop()

info = prep['model_info']
cats = dict(zip(prep['encoder'].feature_names_in_, prep['encoder'].categories_))

st.title('📞 Telco Customer Churn Predictor')
st.write(f"Enter a customer's details to estimate the probability that they will leave. "
         f"Model in use: **{info['model_name']}** (trained on the IBM Telco Customer Churn dataset).")

with st.sidebar:
    st.header('Settings')
    threshold = st.slider('Decision threshold for "will churn"', 0.05, 0.95, 0.50, 0.05,
                          help='Lower it to catch more churners (higher recall, more false alarms).')
    st.caption('Default 0.50 is the threshold used when the model was evaluated in the notebook.')

tab_single, tab_batch, tab_about = st.tabs(['Single customer', 'Batch (CSV upload)', 'About the model'])

# ------------------------------------------------------------------ single customer
with tab_single:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader('Demographics')
        gender = st.selectbox('Gender', cats['gender'])
        senior = st.selectbox('Senior citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', cats['Partner'])
        dependents = st.selectbox('Dependents', cats['Dependents'])
    with c2:
        st.subheader('Account')
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        contract = st.selectbox('Contract', cats['Contract'])
        paperless = st.selectbox('Paperless billing', cats['PaperlessBilling'])
        payment = st.selectbox('Payment method', cats['PaymentMethod'])
        monthly = st.number_input('Monthly charges', 0.0, 200.0, 70.0, 0.05)
        auto_total = st.checkbox('Estimate total charges (tenure × monthly)', value=True)
        total = round(tenure * monthly, 2) if auto_total else st.number_input('Total charges', 0.0, 10000.0, 1000.0)
        if auto_total:
            st.caption(f'Total charges used: {total:,.2f}')
    with c3:
        st.subheader('Services')
        phone = st.selectbox('Phone service', cats['PhoneService'])
        lines = st.selectbox('Multiple lines', cats['MultipleLines'])
        internet = st.selectbox('Internet service', cats['InternetService'])
        security = st.selectbox('Online security', cats['OnlineSecurity'])
        backup = st.selectbox('Online backup', cats['OnlineBackup'])
        protection = st.selectbox('Device protection', cats['DeviceProtection'])
        support = st.selectbox('Tech support', cats['TechSupport'])
        tv = st.selectbox('Streaming TV', cats['StreamingTV'])
        movies = st.selectbox('Streaming movies', cats['StreamingMovies'])

    if st.button('Predict churn', type='primary'):
        row = pd.DataFrame([{
            'gender': gender, 'SeniorCitizen': 1 if senior == 'Yes' else 0, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone, 'MultipleLines': lines,
            'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': protection, 'TechSupport': support, 'StreamingTV': tv,
            'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total}])
        p = float(inference.predict_churn(row, model, prep)[0])
        st.divider()
        left, right = st.columns([1, 2])
        left.metric('Churn probability', f'{p:.1%}')
        if p >= threshold:
            right.error(f'⚠️ Likely to CHURN (probability {p:.1%} ≥ threshold {threshold:.0%}).')
        else:
            right.success(f'✅ Likely to STAY (probability {p:.1%} < threshold {threshold:.0%}).')
        right.progress(min(max(p, 0.0), 1.0))

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write('Upload a CSV with the same columns as the Telco dataset (`customerID` and `Churn` are optional '
             'and ignored). Predictions are added as new columns.')
    st.caption('Required columns: ' + ', '.join(prep['raw_columns']))
    file = st.file_uploader('CSV file', type='csv')
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in prep['raw_columns'] if c not in data.columns]
        if missing:
            st.error(f'Missing required column(s): {", ".join(missing)}')
        else:
            probs = inference.predict_churn(data, model, prep)
            out = data.copy()
            out['Churn probability'] = probs.round(4)
            out['Prediction'] = ['Churn' if x >= threshold else 'Stay' for x in probs]
            st.dataframe(out, use_container_width=True)
            st.download_button('Download predictions', out.to_csv(index=False).encode(), 'churn_predictions.csv')

# ------------------------------------------------------------------ about
with tab_about:
    st.subheader(f"Selected model: {info['model_name']}")
    cols = st.columns(5)
    for col, key in zip(cols, ['Accuracy', 'Precision', 'Recall', 'F1 (weighted)', 'ROC-AUC']):
        col.metric(key, f'{info[key]:.3f}')
    st.caption('Metrics on the held-out 20% test split from the notebook.')
    path = inference.MODELS_DIR / 'model_comparison.csv'
    if path.exists():
        st.subheader('All 10 algorithms compared')
        st.dataframe(pd.read_csv(path), hide_index=True)
