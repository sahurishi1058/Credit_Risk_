import pandas as pd

def log_risky_applicants(df, predictions, threshold=0.5):
    risky_indices = predictions < threshold
    risky_log = df[risky_indices].copy()
    risky_log['Risk_Probability'] = predictions[risky_indices]
    return risky_log
