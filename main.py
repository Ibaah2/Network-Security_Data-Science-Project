from NetworkSecurity.components.data_ingestion import DataIngestion
from NetworkSecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig    
from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
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
        # Initialize the training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        # Initialize the data ingestion configuration
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        # Create an instance of DataIngestion
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        # Extract data and get the artifact
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)