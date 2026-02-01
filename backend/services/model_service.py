"""
Service for machine learning model operations
"""
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.settings import MODEL_DIR
import statsmodels.api as sm

class ModelService:
    """Service for handling ML models and predictions"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        # Mapping: Display Name -> Filename
        self.model_mapping = {
            "Model Global Logistic Regression": "model_logistik_global.pkl",
            "Model Geographically Weighted Logistic Regression": "gwlr_model.pkl",
            "Model Geographically Weighted Logistic Regression Semiparametric": "mgwlr_model.pkl"
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model display names"""
        return list(self.model_mapping.keys())
    
    def get_model_filename(self, display_name: str) -> str:
        """Get filename from display name"""
        return self.model_mapping.get(display_name)
    
    def load_model(self, model_identifier: str) -> Any:
        """
        Load a model from disk
        
        Args:
            model_identifier: Display name or filename
            
        Returns:
            Loaded model object
        """
        # Resolve filename if display name is passed
        filename = self.model_mapping.get(model_identifier, model_identifier)
        
        if filename in self.models:
            return self.models[filename]
            
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.models[filename] = model
                print(f"✓ Loaded model: {filename}")
                return model
        except Exception as e:
            raise RuntimeError(f"Error loading model {filename}: {e}")

    def predict(self, df: pd.DataFrame, model_name: str) -> Tuple[pd.Series, pd.Series]:
        """
        Run prediction on data using specified model
        
        Args:
            df: DataFrame containing features
            model_name: Name of model to use
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        model = self.load_model(model_name)
        
        # Prepare data based on model type
        filename = self.model_mapping.get(model_name, model_name)
        
        if filename == "model_logistik_global.pkl":
            # Specific logic for Global Logistic Regression
            feature_cols = ['DepRatio', 'UMK', 'Industri', 'TPT', 'RumahLayak', 'Sanitasi']
            
            # Check if columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for this model: {', '.join(missing_cols)}")
            
            X = df[feature_cols].copy()
            X = sm.add_constant(X)
        else:
            # Default fallback for other models (GWLR/MGWLR)
            # Assuming they might handle selection or we use all numeric columns minus target
            X = df.select_dtypes(include=[np.number]).dropna()
            if 'p1_encoded' in X.columns:
                 X = X.drop(columns=['p1_encoded'])
        
        try:
            # Check for predict_proba method (sklearn style)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                # Take probability of positive class (usually index 1)
                prob_values = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                preds = model.predict(X)
            
            # Check for statsmodels prediction (returns probabilities by default for Logit)
            elif hasattr(model, 'predict'):
                # For statsmodels Logit, predict() returns probabilities
                raw_prediction = model.predict(X)
                
                # Force conversion to numeric and handle potential Series/ndarray issues
                try:
                    # Ensure it's a pandas Series
                    if not isinstance(raw_prediction, pd.Series):
                        raw_prediction = pd.Series(raw_prediction, index=X.index)
                    
                    # Convert to numeric, coercing errors
                    prob_values = pd.to_numeric(raw_prediction, errors='coerce').fillna(0)
                    
                    # Check if values are effectively probabilities (mostly between 0 and 1)
                    # We assume statsmodels predict() returns probabilities for Logit
                    # If max value > 1.5, it might be linear predictor, but default is probability.
                    
                    # Convert to class (threshold 0.5)
                    preds = (prob_values >= 0.5).astype(int)
                    
                except Exception as conversion_error:
                     print(f"Warning: Could not convert statsmodels output to probability: {conversion_error}")
                     # Fallback
                     preds = raw_prediction
                     prob_values = np.zeros(len(preds))

            else:
                 # Check if it's a statsmodels result or similar wrapper
                if hasattr(model, 'prediksi'): # Custom method name?
                     preds = model.prediksi(X)
                     prob_values = np.zeros(len(preds))
                else:
                    raise AttributeError("Model does not have predictable method")
            
            return pd.Series(preds, index=X.index), pd.Series(prob_values, index=X.index)
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculate classification metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

# Global instance
model_service = ModelService()
