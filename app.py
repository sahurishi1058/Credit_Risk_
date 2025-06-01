# app/app.py
import streamlit as st
import pandas as pd
from model_inference import load_model, predict
from explainability import get_shap_summary
from Credit_Risk_prediction.logger import log_risky_applicants
from streamlit_shap import st_shap
import shap

st.set_page_config("Credit Risk Predictor", layout="wide")
st.title("🏦 Credit Risk Prediction App")
model_bundle = load_model(r"C:\Users\sahur\OneDrive\Desktop1\Projects\Credit_Risk_prediction\Credit_Risk_prediction\models\credit_model.pkl")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 Input Sample:")
    st.dataframe(df.head())

    if st.button("Predict"):
        y_pred, y_prob = predict(model_bundle, df)
        df['Default_Prediction'] = y_pred
        df['Default_Probability'] = y_prob
        risky_log = log_risky_applicants(df, y_prob, threshold=0.5)
        approved= df[df['Default_Prediction'] == 0]
        rejected = df[df['Default_Prediction'] == 1]
        st.success(f"✅ Approved: {len(approved)}")
        st.subheader("✅ Approved Applicant Log")
        st.dataframe(approved)
        st.write(f"🔴 Rejected Applicants: {len(rejected)}")
        st.subheader("🚨 Risky Applicant Log")
        df = rejected.copy()
        df = df.drop(columns=['Default_Prediction', 'Default_Probability'], errors='ignore')
        df = df.reset_index(drop=True)
        st.dataframe(df.head())
        csv = risky_log.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Risky Applicants Log", data=csv, file_name="risky_applicants.csv", mime="text/csv")
        st.success("✅ Prediction done.")
        

      
    if st.checkbox("Show SHAP Explainability"):
        model = model_bundle['model']
        preprocessor = model_bundle['preprocessor']
        X_sample = preprocessor.transform(df)

        with st.spinner("Generating SHAP plot..."):
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)

            # Display SHAP summary plot
            st_shap(shap.summary_plot(shap_values, X_sample))
