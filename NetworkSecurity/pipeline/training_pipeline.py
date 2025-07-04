import os 
import sys

from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.components.data_ingestion import DataIngestion
from NetworkSecurity.components.data_validation import DataValidation
from NetworkSecurity.components.data_transformation import DataTransformation
from NetworkSecurity.components.model_trainer import ModelTrainer

# Load config entities
from NetworkSecurity.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,\
                                                DataTransformationConfig, ModelTrainerConfig
# Load artifacts
from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
                                                DataTransformationArtifact, ModelTrainerArtifact
from NetworkSecurity.cloud.s3_syncer import s3sync

from dotenv import load_dotenv
import os
import sys

load_dotenv()  # Loads variables from .env text file

TRAINING_BUCKET_NAME = os.getenv("BUCKET_NAME")


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3sync = s3sync()

    ########################
    #Data Ingestion
    ########################    

    def start_data_ingestion(self):
        try:  
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            # Create an instance of DataIngestion
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            # Extract data and get the artifact
            logging.info("Initiating data ingestion...")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed")

            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    ################################
    # Data Validation
    ################################
    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            # Create an instance of DataValidation
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                            data_validation_config=data_validation_config)
            # Initialize the data validation configuration
            logging.info("Initializing data validation configuration...")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed successfully.")

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    ################################
    # Data Transformation
    ################################
    def start_data_transformation(self, data_validation_artifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            # Create an instance of DataTransformation
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                    data_transformation_config=data_transformation_config)
            # Initiate the data transformation process
            logging.info("Initiating data transformation...")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed successfully.")

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    ################################
    # Model Training, Evaluation, and Hyperparameter Tuning
    ################################
    def start_model_training(self, data_transformation_artifact):
        try:
            logging.info("Starting model training...")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                        data_model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed successfully.")

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f's3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}'
            self.s3sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,
                                          aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def sync_model_dir_to_s3(self):
        try:
            local_model_dir = os.path.abspath('final_model')
            aws_bucket_url = f's3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}'
            self.s3sync.sync_folder_to_s3(folder = local_model_dir,
                                          aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    ##################################################
    # We will now create a function which will run the data ingestion, validation, transformation, & training
    ##################################################
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact)

            self.sync_artifact_dir_to_s3()
            self.sync_model_dir_to_s3()

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
