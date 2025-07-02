from NetworkSecurity.constants.training_pipeline import SAVE_MODEL_DIR, MODEL_FILE_NAME
import os
import sys

from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self, X):
        """
        Predicts the target variable using the preprocessor and model.
        
        Args:
            X (array-like): Input features.
        
        Returns:
            array: Predicted values.
        """
        try:
            logging.info("Making predictions.")
            X_transform = self.preprocessor.transform(X)
            y_hat = self.model.predict(X_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)