from data_loader import load_data, preprocess_data, split_data
from model_train import train_model, evaluate, save_model

# Load and preprocess data
df = load_data(r"C:\Users\sahur\OneDrive\Desktop\Projects\Credit_Risk_prediction\datasets\application_train.csv")
X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
evaluate(X_test, y_test, model)

# Save model and preprocessor
save_model(model,preprocessor, "models/credit_model.pkl")  # Only save model

