import shap
from streamlit_shap import st_shap

def get_shap_summary(model, x_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(x_sample)

    # Use streamlit-shap to render the SHAP plot in Streamlit
    st_shap(shap.summary_plot(shap_values, x_sample))
