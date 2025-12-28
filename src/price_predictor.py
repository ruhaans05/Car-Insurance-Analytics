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

# Generate synthetic insurance prices based on risk factors
# Base price: $500
# Adjustments based on various risk factors
base_price = 500

# Risk multipliers
age_mult = 1 + (3 - df['age']) * 0.15  # Younger = higher risk
income_mult = 1 + (2 - df['income']) * 0.20  # Lower income = higher premium
credit_mult = 1 + (1 - df['credit_score']) * 0.30  # Lower credit = higher premium
driving_exp_mult = 1 + (3 - df['driving_experience']) * 0.20  # Less experience = higher
violations_mult = 1 + df['speeding_violations'] * 0.15  # Each violation adds 15%
duis_mult = 1 + df['DUIs'] * 0.40  # Each DUI adds 40%
accidents_mult = 1 + df['past_accidents'] * 0.25  # Each accident adds 25%

# Combine all factors
price = (base_price * age_mult * income_mult * credit_mult * 
         driving_exp_mult * violations_mult * duis_mult * accidents_mult)

# Add some randomness for realism
np.random.seed(42)
price = price * (1 + np.random.normal(0, 0.1, len(price)))  # ±10% noise
price = np.clip(price, 300, 5000)  # Cap between $300-$5000

# LOG TRANSFORMATION: Critical for insurance pricing
y = np.log1p(price)
print(f"Generated synthetic prices. Range: ${price.min():.2f} - ${price.max():.2f}")
print("Using synthetic price as target (log-transformed).")

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