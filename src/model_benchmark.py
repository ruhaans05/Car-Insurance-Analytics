import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def prepare_comparison_data(path="Data/customer-data.csv"):
    """
    Load and preprocess data for model comparison.
    """
    df = pd.read_csv(path)
    
    # Simple encoding for categorical variables
    if 'gender' in df.columns:
        df['gender'] = LabelEncoder().fit_transform(df['gender'])
    if 'race' in df.columns:
        df['race'] = LabelEncoder().fit_transform(df['race'])
        
    # We will predict 'driving_experience' as a proxy for risk if 
    # specific 'premium' or 'claim' columns aren't available yet.
    target = 'driving_experience' 
    
    # Define Feature Sets
    all_features = ['age', 'gender', 'race', 'income']
    safe_features = ['age', 'gender'] # Based on your rejection hypothesis
    
    return df, all_features, safe_features, target

def train_and_evaluate(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return mae, r2

if __name__ == "__main__":
    df, all_feats, safe_feats, target_col = prepare_comparison_data()
    y = df[target_col]

    # 1. Evaluate "Aware" Model (Includes Race/Income)
    mae_aware, r2_aware = train_and_evaluate("Aware Model", df[all_feats], y)

    # 2. Evaluate "Unaware" Model (Only Age/Gender)
    mae_unaware, r2_unaware = train_and_evaluate("Unaware Model", df[safe_feats], y)

    # Output Results
    print("--- Model Comparison Results ---")
    print(f"Aware Model (All):   MAE = {mae_aware:.2f}, R2 = {r2_aware:.4f}")
    print(f"Unaware Model (Safe): MAE = {mae_unaware:.2f}, R2 = {r2_unaware:.4f}")
    
    lift = ((mae_unaware - mae_aware) / mae_aware) * 100
    print(f"\nAccuracy Loss by removing sensitive columns: {lift:.2f}%")
    
    if lift < 1.0:
        print("CONCLUSION: Accept Hypothesis. Sensitive columns provide no significant value.")
    else:
        print("CONCLUSION: Reject Hypothesis. Columns provide value, but consider legal constraints.")