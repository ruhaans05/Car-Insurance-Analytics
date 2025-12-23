import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers


def build_stable_model(input_shape):
    """
    DNN with BatchNormalization to handle outliers like raw credit scores.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(), 
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        # Using linear activation because we are predicting log(price)
        layers.Dense(1, activation='linear') 
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model

# 1. Load Data
df = pd.read_csv("Data/cleaned_customer_data.csv")

# 2. Prepare Features (X) and Target (y)
# We drop price (target), outcome (redundant), and id from features
cols_to_drop = ['price', 'outcome', 'id']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Determine target column robustly. Prefer `price` (continuous),
# fall back to `outcome` (binary). Provide a clear error if neither.
if 'price' in df.columns:
    # LOG TRANSFORMATION: Critical for insurance pricing
    y = np.log1p(df['price'])
    print("Using 'price' column as target (log-transformed).")
elif 'outcome' in df.columns:
    # Binary target — do not log-transform
    y = df['outcome']
    print("'price' column not found; using 'outcome' column as target (no log).")
else:
    raise KeyError("No suitable target column found. Expected 'price' or 'outcome' in Data/cleaned_customer_data.csv")

# 3. Scaling and Saving
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
os.makedirs("Models", exist_ok=True)
joblib.dump(scaler, "Models/scaler.bin")

# 4. Train
model = build_stable_model(X_scaled.shape[1])
print(f"Training on {X_scaled.shape[1]} features...")
model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)

# 5. Save Model
model.save("Models/insurance_pricing_v1.h5")
print("Model and Scaler saved. Now run get_predicted_price.py")