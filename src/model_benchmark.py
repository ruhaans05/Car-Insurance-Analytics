import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

def prepare_comparison_data(path="Data/customer-data.csv"):
    """
    Load and preprocess data specifically for the car insurance benchmarking.
    """
    df = pd.read_csv(path)
    
    # 1. Map Ordinal Ranges to Numbers (matches your Data Sample)
    # This is better than LabelEncoder because it preserves the 'order' of age/income
    age_map = {'16-25': 0, '26-39': 1, '40-64': 2, '65+': 3}
    income_map = {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3}
    exp_map = {'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3}
    
    df['age'] = df['age'].map(age_map)
    df['income'] = df['income'].map(income_map)
    df['driving_experience'] = df['driving_experience'].map(exp_map)
    
    # 2. Encode categorical/binary columns
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['race'] = le.fit_transform(df['race'])
    
    # Target: 'outcome' (True = Claim, False = No Claim)
    df['outcome'] = df['outcome'].astype(int)
    target = 'outcome'
    
    # 3. Define Feature Sets for Hypothesis Testing
    # 'Aware' includes sensitive data; 'Safe' excludes it.
    all_features = [
        'age', 'gender', 'race', 'driving_experience', 
        'income', 'credit_score', 'annual_mileage'
    ]
    safe_features = [
        'age', 'gender', 'driving_experience', 
        'credit_score', 'annual_mileage'
    ]
    
    return df.dropna(), all_features, safe_features, target

def train_and_evaluate(X, y):
    """
    Trains an XGBoost Classifier and returns key performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We use Classifier because 'outcome' is binary (0 or 1)
    model = xgb.XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get probability predictions for AUC/Log-Loss
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    loss = log_loss(y_test, probs)
    
    return auc, loss

if __name__ == "__main__":
    df, all_feats, safe_feats, target_col = prepare_comparison_data()
    y = df[target_col]

    # 1. Evaluate "Aware" Model (Includes Race and Income)
    auc_aware, loss_aware = train_and_evaluate(df[all_feats], y)

    # 2. Evaluate "Unaware" Model (Excludes Race and Income)
    auc_unaware, loss_unaware = train_and_evaluate(df[safe_feats], y)

    # Output Results
    print("\n" + "="*40)
    print("INSURANCE PRICING BENCHMARK RESULTS")
    print("="*40)
    print(f"Aware Model (Sensitive):   AUC = {auc_aware:.4f}, LogLoss = {loss_aware:.4f}")
    print(f"Unaware Model (Safe):      AUC = {auc_unaware:.4f}, LogLoss = {loss_unaware:.4f}")
    
    # Calculate performance degradation
    # A drop in AUC means the model is less able to distinguish between high and low risk.
    auc_drop = ((auc_aware - auc_unaware) / auc_aware) * 100
    
    print("-" * 40)
    print(f"Predictive Power Loss: {auc_drop:.2f}%")
    
    if auc_drop < 1.0:
        print("\nCONCLUSION: Accept Hypothesis.")
        print("Sensitive columns provide negligible lift. Recommend removal to mitigate ethical risk.")
    else:
        print("\nCONCLUSION: Reject Hypothesis.")
        print("Sensitive columns provide unique signal. Policy review required for implementation.")