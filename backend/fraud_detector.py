# backend/fraud_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # <-- Import joblib
import os      # <-- Import os to build file paths

class EthereumFraudDetector:
    def __init__(self, model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
        """
        Load the pre-trained model and scaler from disk.
        """
        # Build absolute paths to ensure they are found
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_file = os.path.join(base_dir, model_path)
        scaler_file = os.path.join(base_dir, scaler_path)

        # ✅ SOLUTION: Load your TRAINED model and FITTED scaler
        try:
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            print("✅ Successfully loaded pre-trained model and scaler.")
        except FileNotFoundError as e:
            print(f"🚨 FATAL ERROR: Model/Scaler file not found.")
            print(f"Tried to load: {model_file} and {scaler_file}")
            print(f"Make sure 'model.pkl' and 'scaler.pkl' are in a 'models' directory next to this file.")
            raise e
        except Exception as e:
            print(f"🚨 FATAL ERROR: Could not load model or scaler. {e}")
            raise e

    def preprocess_single_transaction(self, transaction_data):
        """Preprocess single transaction data"""
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = pd.DataFrame(transaction_data)
            
        # Feature engineering (same as training)
        df['Value_to_Fee_Ratio'] = df['Transaction_Value'] / (df['Transaction_Fees'] + 1e-8)
        df['Input_Output_Ratio'] = df['Number_of_Inputs'] / (df['Number_of_Outputs'] + 1e-8)
        df['Gas_Efficiency'] = df['Transaction_Value'] / (df['Gas_Price'] + 1e-8)
        df['Balance_Turnover'] = df['Transaction_Value'] / (df['Wallet_Balance'] + 1e-8)
        
        return df
    
    def predict_fraud_risk(self, transaction_data):
        """Predict fraud risk for a transaction"""
        try:
            df = self.preprocess_single_transaction(transaction_data)
            
            feature_columns = [
                'Transaction_Value', 'Transaction_Fees', 'Number_of_Inputs', 
                'Number_of_Outputs', 'Gas_Price', 'Wallet_Age_Days', 
                'Wallet_Balance', 'Transaction_Velocity', 'Exchange_Rate',
                'Value_to_Fee_Ratio', 'Input_Output_Ratio', 
                'Gas_Efficiency', 'Balance_Turnover'
            ]
            
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            X = df[feature_columns].fillna(0)
            
            # Use the LOADED scaler
            X_scaled = self.scaler.transform(X)
            
            # Get prediction probability from the LOADED model
            risk_score = self.model.predict_proba(X_scaled)[0][1]
            
            return float(risk_score)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.5