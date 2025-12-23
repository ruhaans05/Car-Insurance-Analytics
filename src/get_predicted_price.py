import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from thefuzz import process


class InsurancePredictor:
    def __init__(self, model_path="Models/insurance_pricing_v1.h5", scaler_path="Models/scaler.bin"):
        # Fix for the 'mse' serialization issue in newer Keras/TF versions
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found. Please run your training script first.")
            
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.scaler = joblib.load(scaler_path)
        
        # Consistent mapping used across the entire project
        self.mappings = {
            'age': {'16-25': 0, '26-39': 1, '40-64': 2, '65+': 3},
            'gender': {'female': 0, 'male': 1},
            'race': {'majority': 0, 'minority': 1},
            'driving_experience': {'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3},
            'education': {'none': 0, 'high school': 1, 'university': 2},
            'income': {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3},
            'vehicle_year': {'before 2015': 0, 'after 2015': 1},
            'vehicle_ownership': {'false': 0, 'true': 1, 'no': 0, 'yes': 1},
            'married': {'false': 0, 'true': 1, 'no': 0, 'yes': 1},
            'children': {'false': 0, 'true': 1, 'no': 0, 'yes': 1}
        }

    def get_user_input(self):
        print("\n" + "="*45)
        print("   CAR INSURANCE ANALYTICS: PRICE PREDICTOR   ")
        print("="*45)
        
        user_data = {}
        # Iterate through features the model was actually trained on
        for feature in self.scaler.feature_names_in_:
            display_name = feature.replace('_', ' ').title()
            
            if feature in self.mappings:
                options = list(self.mappings[feature].keys())
                val = input(f"{display_name} ({', '.join(options)}): ").lower().strip()
                
                # Fuzzy matching to prevent crashes on typos
                match, score = process.extractOne(val, options)
                if score < 60:
                    print(f"   [!] Uncertain input. Defaulting to '{match}'")
                user_data[feature] = match
            else:
                val = input(f"{display_name}: ")
                num_val = float(val) if val else 0.0
                
                # Handle Credit Score scale (e.g., 600 -> 0.70)
                if feature == 'credit_score' and num_val > 1.0:
                    num_val = num_val / 850.0 
                user_data[feature] = num_val
                
        return user_data

    def predict(self, customer_data):
        df = pd.DataFrame([customer_data])
        
        # Apply integer mappings
        for col, mapping in self.mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Ensure column order matches the Scaler
        X = df[self.scaler.feature_names_in_].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)
        
        # 1. Prediction comes out in Log Scale
        prediction_log = self.model.predict(X_scaled, verbose=0)
        
        # 2. Inverse Log: Convert log(price) -> actual dollars
        # np.expm1(x) is e^x - 1
        final_price = float(np.expm1(prediction_log[0][0]))
        return final_price

if __name__ == "__main__":
    predictor = InsurancePredictor()
    try:
        customer_dict = predictor.get_user_input()
        price = predictor.predict(customer_dict)
        
        print("\n" + "-"*45)
        # We apply a $500 floor because base insurance costs have a minimum threshold
        print(f"ESTIMATED ANNUAL PREMIUM: ${max(price, 500.00):,.2f}")
        print("-" * 45 + "\n")
    except Exception as e:
        print(f"\n[System Error]: {e}")