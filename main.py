from NetworkSecurity.components.data_ingestion import DataIngestion
from NetworkSecurity.components.data_validation import DataValidation
from NetworkSecurity.components.data_transformation import DataTransformation
from NetworkSecurity.entity.config_entity import DataValidationConfig
from NetworkSecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig    
from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.components.model_trainer import ModelTrainer
from NetworkSecurity.entity.config_entity import ModelTrainerConfig
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pymongo
import certifi
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")  
cluster = os.getenv("MONGO_CLUSTER")

uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"


if __name__ == "__main__":
    """Main entry point for the data ingestion process.
    Initializes the training pipeline configuration, data ingestion configuration,
    and the NetworkDataExtraction component to extract data from MongoDB and convert it to a DataFrame.
    """
    logging.info("Starting the data ingestion process...")

    try:
        ################################
        # Data Ingestion
        ################################

        # Initialize the training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        # Initialize the data ingestion configuration
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        # Create an instance of DataIngestion
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        # Extract data and get the artifact
        logging.info("Initiating data ingestion...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion Completed")

        ################################
        # Data Validation
        ################################

        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        # Create an instance of DataValidation
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                          data_validation_config=data_validation_config)
        # Initialize the data validation configuration
        logging.info("Initializing data validation configuration...")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully.")

        ################################
        # Data Transformation
        ################################
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        # Create an instance of DataTransformation
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                  data_transformation_config=data_transformation_config)
        # Initiate the data transformation process
        logging.info("Initiating data transformation...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully.")

        ################################
        # Model Training, Evaluation, and Hyperparameter Tuning
        ################################
        logging.info("Starting model training...")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                       data_model_trainer_config=model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed successfully.")       
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)