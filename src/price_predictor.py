import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def build_pricing_model(input_shape):
    """
    Creates a Deep Neural Network for regression.
    """
    model = tf.keras.Sequential([
        # Input Layer
        layers.Input(shape=(input_shape,)),
        
        # Hidden Layers with Dropout to prevent overfitting on sensitive data
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        
        # Output Layer: Single value for the price
        # Using 'softplus' ensures the price is always positive
        layers.Dense(1, activation='softplus') 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse', # Mean Squared Error
        metrics=['mae'] # Mean Absolute Error
    )
    return model

if __name__ == "__main__":
    # 1. Load your cleaned data
    # Ensure you are using the 'cleaned' version from your Main.py pipeline
    if not os.path.exists("Data/cleaned_customer_data.csv"):
        print("Run Main.py first to generate the cleaned dataset.")
    else:
        df = pd.read_csv("Data/cleaned_customer_data.csv")

        # 2. Define Features and Target
        # If 'price' isn't in your data yet, you can use 'outcome' as a proxy 
        # or create a synthetic 'premium' column for testing.
        X = df.drop(columns=['outcome']) # All metrics
        y = df['outcome'] # Or your price column

        # 3. Scale the data (Crucial for Neural Networks)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 4. Initialize and Train
        model = build_pricing_model(X_train.shape[1])
        
        print("\n--- Training TensorFlow Pricing Model ---")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # 5. Evaluate
        results = model.evaluate(X_test, y_test)
        print(f"\nModel Performance: MAE = {results[1]:.4f}")

        # 6. Save Model
        model.save("Models/insurance_pricing_v1.h5")
        print("Model saved to Models/insurance_pricing_v1.h5")