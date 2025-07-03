import sys
import os
import pandas as pd
import numpy as np

from NetworkSecurity.constants.training_pipeline import TARGET_COLUMN
from NetworkSecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurity.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from NetworkSecurity.entity.config_entity import TrainingPipelineConfig
from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
                                    DataTransformationArtifact, ClassificationMetricArtifact, \
                                    ModelTrainerArtifact
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.utils.main_utils.utils import save_object, save_numpy_array_data, load_object, \
                                                    load_numpy_array_data, evaluate_models
from NetworkSecurity.utils.ml_utils.metrics.classification_metric import get_classification_score
from NetworkSecurity.utils.ml_utils.model.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

import mlflow
from mlflow.models.signature import infer_signature

class ModelTrainer:
    """
    This class is responsible for training the machine learning model.
    It handles the loading of data, training the model, and saving the trained model.
    """

    def __init__(self, data_model_trainer_config: ModelTrainerConfig,
                        data_transformation_artifact: DataTransformationArtifact):
        """
        Class constructor that initializes the model trainer with configuration and data transformation artifact.
        Args:
            data_model_trainer_config (ModelTrainerConfig): Configuration object containing model training details.
            data_transformation_artifact (DataTransformationArtifact): Artifact containing transformed data.
        """
        try:
            self.data_model_trainer_config = data_model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, X_train, best_model, classification_metric):
        with mlflow.start_run():
            accuracy=classification_metric.accuracy
            precision=classification_metric.precision
            recall=classification_metric.recall
            f1_score=classification_metric.f1_score
            roc_auc=classification_metric.roc_auc

            # Log metrics
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1_score)
            mlflow.log_metric('roc_auc', roc_auc)

            # Log best model
            signature = infer_signature(X_train, best_model.predict(X_train))
            input_example = X_train[:1]
            mlflow.sklearn.log_model(sk_model=best_model, name='model',
                                     input_example=input_example,
                                     signature=signature)
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Trains the machine learning model using the provided training data.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        
        Returns:
            The trained model.
        """
        try:
            models = {
                #"LogisticRegression": LogisticRegression(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "KNeighborsClassifier": KNeighborsClassifier()
            }

            params = {
                #"LogisticRegression": {"C": [1.0], "max_iter": [10, 20, 50, 100, 100]},
                "DecisionTreeClassifier": {"max_depth": [5, 10]},
                "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [5, 10]},
                "GradientBoostingClassifier": {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 1.0]},
                "AdaBoostClassifier": {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 1.0]},
                "XGBClassifier": {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 1.0]},
                "KNeighborsClassifier": {"n_neighbors": [3, 5, 10]}
            }

            # Evaluate models and select the best one
            best_model_name, best_model, model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            logging.info(f"Model report: {model_report}")

            # Get the best model score
            best_model_score = model_report[best_model_name]

            # Get the best model name
            best_model_name = best_model_name
            best_model = models[best_model_name]

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate classification metrics for test and train data
            logging.info("Calculating classification metrics for train and test data.")
            train_classifcation_metrics = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )
            test_classification_metrics = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )
            logging.info(f"Train classification metrics: {train_classifcation_metrics}")
            logging.info(f"Test classification metrics: {test_classification_metrics}")

            # Track the experiemnts with MLFlow
            # On Train
            self.track_mlflow(X_train, best_model, train_classifcation_metrics)
            # On Test
            self.track_mlflow(X_test, best_model, test_classification_metrics)

            processor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.join(
                self.data_model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Network_Model = NetworkModel(preprocessor=processor, model=best_model)
            save_object(
                file_path=self.data_model_trainer_config.trained_model_file_path,
                obj=Network_Model
            )

            # Create a model artifact

            ModelTrainerArtifact(
                trained_model_file_path=self.data_model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_classifcation_metrics,
                test_metric_artifact=test_classification_metrics,
                best_model_name = best_model_name
            )
            return ModelTrainerArtifact    
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.
        
        Args:
            training_pipeline_config (TrainingPipelineConfig): Configuration for the training pipeline.
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing ingested data.
            data_validation_artifact (DataValidationArtifact): Artifact containing validated data.
        
        Returns:
            ModelTrainerArtifact: Artifact containing trained model and metrics.
        """
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Loading training and testing data.")
            # Load training and testing data
            train_data = load_numpy_array_data(file_path=train_file_path)
            test_data = load_numpy_array_data(file_path=test_file_path)

            # Load the training and testing features and labels
            X_train, y_train, X_test, y_test = (
                train_data[:, :-1],
                train_data[:, -1], 
                test_data[:, :-1], 
                test_data[:, -1]
            )
            logging.info("Training and testing data loaded successfully.")

            model_trainer_artifact = self.train_model(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test
            )
            logging.info("Model training completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)