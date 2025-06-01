import joblib,os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model= XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model
def evaluate(x_test, y_test, model):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
def save_model(model, preprocessor=None, model_path="models/credit_model_bundle.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_bundle = {
        'model': model,
        'preprocessor': preprocessor
    }
    joblib.dump(model_bundle, model_path)
    print(f"Model bundle saved to {model_path}")
