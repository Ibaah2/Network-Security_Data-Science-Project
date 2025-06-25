from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.entity.config_entity import DataIngestionConfig
from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
import certifi
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")

uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
mongo_client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))
#MONGO_DB_URL = os.getenv('MONGO_DB_URL')

class DataIngestion:
    """
    This class handles the ingestion of network data into MongoDB.
    It reads data from a CSV file, converts it to JSON, and inserts it into the database.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Class constructor. Initializes the MongoDB client and sets up the database and collection.      
        Args:
            data_ingestion_config (DataIngestionConfig): Configuration object containing database and collection details.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            
        except Exception as e:  
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self, uri):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))
            collection = self.mongo_client[database_name][collection_name]
            logging.info(f"Exporting collection {collection_name} from database {database_name} as a DataFrame.")
            
            df = pd.DataFrame(list(collection.find()))

            if '_id' in df.columns:
                df.drop('_id', axis=1, inplace=True)
            
            df.replace(['na', 'NA', 'NaN', 'nan', 'N/A', 'n/a'], np.nan, inplace=True)
            
            return df
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)    
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Exports the DataFrame to a feature store file.
        Args:
            df (pd.DataFrame): DataFrame to be exported.
        """
        try:
            logging.info(f"Exporting data to feature store at {self.data_ingestion_config.feature_store_file_path}.")

            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)

            return dataframe
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)       
            
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the DataFrame into training and testing sets.
        Args:
            dataframe (pd.DataFrame): DataFrame to be split.
        Returns:
            tuple: Training and testing DataFrames.
        """
        try:
            logging.info("Splitting data into training and testing sets.")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info(f"Training set size: {len(train_set)}, Testing set size: {len(test_set)}")

            # Ensure the directories exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.testing_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Data split into training and testing sets successfully.")

            return train_set, test_set

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_ingestion(self):
        """
        Reads data from a CSV file and returns it as a pandas DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the network data.
        """
        try:
            logging.info("Starting data ingestion process.")

            uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"

            dataframe = self.export_collection_as_dataframe(uri)
            dataframe = self.export_data_into_feature_store(dataframe)

            train_set, test_set = self.split_data_as_train_test(dataframe)

            # Create and return the DataIngestionArtifact
                
            data_ingestion_artifact = DataIngestionArtifact(                
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path
                )
            #logging.info(f"Data ingestion completed. Training set shape: {train_set.shape}, Testing set shape: {test_set.shape}")

            logging.info("Data ingestion completed successfully.")

            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

            
        

        
        