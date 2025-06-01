import joblib
import pandas as pd
def load_model(model_path):
    return joblib.load(model_path)
def predict(model_bundle, input_data):
    model = model_bundle['model']
    preprocessor = model_bundle['preprocessor']
    print("Preprocessor type:", type(preprocessor))
    print("Preprocessor attributes:", dir(preprocessor))
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    print("Input data columns:", input_data.columns)
    processed = preprocessor.transform(input_data)
    return model.predict(processed), model.predict_proba(processed)[:, 1]