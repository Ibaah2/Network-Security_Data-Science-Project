import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from NetworkSecurity.constants.training_pipeline import TARGET_COLUMN
from NetworkSecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurity.entity.config_entity import DataTransformationConfig
from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
                                    DataTransformationArtifact
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.utils.main_utils.utils import save_object, save_numpy_array_data


class DataTransformation:
    """
    This class handles the transformation of network data.
    It performs data cleaning, feature engineering, and prepares the data for training.
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Class constructor. Initializes the data transformation configuration.
        Args:            
            data_validation_artifact (DataValidationArtifact): Artifact containing the validated data.
            data_transformation_config (DataTransformationConfig): Configuration object containing 
            transformation details.
        """
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            #self._imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            #self._scaler = StandardScaler()
            #self._pipeline = Pipeline(steps=[
                #('imputer', self._imputer),
                #('scaler', self._scaler)
            #])
            logging.info("DataTransformation class initialized successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads data from a CSV file and returns it as a DataFrame.
        
        Args:
            file_path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing the data from the CSV file.
        """
        try:
            logging.info(f"Reading data from {file_path}.")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformation_pipeline(cls) -> Pipeline:
        """
        Returns the data transformation pipeline.

        Args:
            cls (DataTransformation): The DataTransformation instance.
        
        Returns:
            Pipeline: The data transformation pipeline with imputer.
        """
        try:
            logging.info("Creating data transformation pipeline with KNN imputer.")
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            scaler:StandardScaler = StandardScaler()
            processor:Pipeline = Pipeline(steps=[
                ('imputer', imputer),
                ('scaler', scaler)
            ])
            logging.info("Data transformation pipeline created successfully.")
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process.
        
        Returns:
            DataTransformationArtifact: Artifact containing the transformed data and related information.
        
        Raises:
            NetworkSecurityException: If any error occurs during the transformation process.
        """
        try:
            logging.info("Starting data transformation process.")
            # Read the validated data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            
            # Separate features and target variable
            # For training data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis = 1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # For testing data
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis = 1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # Apply the transformation pipeline
            processor_object = self.get_data_transformation_pipeline()
            transformed_input_feature_train = processor_object.fit_transform(input_feature_train_df)
            transformed_input_feature_test = processor_object.transform(input_feature_test_df)
            
            # Combine the transformed features with the target variable
            train_arr = np.c_[transformed_input_feature_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test, np.array(target_feature_test_df)]

            # Save the transformed data to numpy arrays
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, 
                                  array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, 
                                  array=test_arr)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path, 
                        obj=processor_object)
            
            logging.info("Data transformation completed successfully.")

            # Create and return the DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path)
            return data_transformation_artifact                      

        except Exception as e:
            raise NetworkSecurityException(e, sys)