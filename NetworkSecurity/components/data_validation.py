from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from NetworkSecurity.entity.config_entity import DataValidationConfig
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from NetworkSecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
import os
import sys
import pandas as pd
from scipy.stats import ks_2samp


class DataValidation:
    """
    This class handles the validation of network data.
    It checks for data drift and validates the schema of the data.
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Class constructor. Initializes the data validation configuration.
        Args:
            data_validation_config (DataValidationConfig): Configuration object containing validation details.
        """
        try:
            self.schema_file_path = SCHEMA_FILE_PATH            
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)         
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

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates the number of columns in the DataFrame against the schema.
        
        Args:
            dataframe (pd.DataFrame): DataFrame to validate.
        
        Returns:
            bool: True if the number of columns matches the schema, False otherwise.
        """

        try:
            logging.info("Validating number of columns in the DataFrame.")
            number_of_columns = len(self._schema_config)
            logging.info(f"Expected number of columns: {number_of_columns}")
            logging.info(f"Actual number of columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            else:
                logging.error(f"Number of columns mismatch: expected {number_of_columns}, found {len(dataframe.columns)}.")
                return False            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    #def check_numeric_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if the DataFrame contains numeric columns as per the schema.
        
        Args:
            dataframe (pd.DataFrame): DataFrame to check.
        
        Returns:
            bool: True if numeric columns exist, False otherwise.
        """
        #try:
            #logging.info("Checking for numeric columns in the DataFrame.")
            #numeric_columns = [col for col in self._schema_config if self._schema_config[col]['type'] == 'numeric']
            #logging.info(f"Numeric columns defined in schema: {numeric_columns}")
            #if len(numeric_columns) >= 1:
                #logging.info("Numeric columns exist in the DataFrame.")
                #return True
            #else:
                #logging.error("No numeric columns found in the DataFrame as per schema.")
                #return False
        #except Exception as e:
            #raise NetworkSecurityException(e, sys)
        
    def check_data_drift(self, base_df, current_df, threshold = 0.05) -> bool:
        """
        Checks for data drift between training and testing datasets.
        
        Args:
            train_dataframe (pd.DataFrame): Training dataset.
            test_dataframe (pd.DataFrame): Testing dataset.
        
        Returns:
            dict: Dictionary containing drift report.
        """
        try:
            logging.info("Checking for data drift between training and testing datasets.")
            status = True
            drift_report = {}
            for columns in base_df.columns:
                d1 = base_df[columns]
                d2 = current_df[columns]
                sample_dist = ks_2samp(d1, d2)
                if threshold <= sample_dist.pvalue:
                    drift_found = False
                else:
                    drift_found = True
                    status = False
                drift_report.update({columns: {
                    'p-value': float(sample_dist.pvalue),
                    'statistic': float(sample_dist.statistic),
                    'drift_status': drift_found
                }})  
            ## Save drift report to a file
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            
            write_yaml_file(file_path=drift_report_file_path, content=drift_report)                  
            logging.info(f"Data drift report: {drift_report}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process.
        Validates the schema and checks for data drift.
        
        Returns:
            DataValidationArtifact: Artifact containing validation results.
        """
        try:
            #logging.info("Starting data validation process.")
            #self.validate_schema()
            #drift_report = self.check_data_drift()
            #data_validation_artifact = DataValidationArtifact(
                #is_validated=True,
                #message="Data validation completed successfully.",
                #drift_report=drift_report
            #)
            #logging.info("Data validation completed successfully.")

            train_file_path = self.data_ingestion_artifact.training_file_path
            test_file_path = self.data_ingestion_artifact.testing_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Validate schema
            status = self.validate_number_of_columns(train_dataframe) 
            if not status:
                print(f"Data validation failed: Schema mismatch in training data.")
            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                print(f"Data validation failed: Schema mismatch in testing data.")

            # Check for numeric columns
            #status_numeric = self.check_numeric_column_exist(train_dataframe)
            #if not status_numeric:
                #print(f"No numeric columns found in training data as per schema.") 
            #status_numeric = self.check_numeric_column_exist(test_dataframe)
            #if not status_numeric:
                #print(f"No numeric columns found in testing data as per schema.")

            # Check for data drift
            drift_status = self.check_data_drift(base_df=train_dataframe, current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)

            dir_path = os.path.dirname(self.data_validation_config.valid_test_file_path)
            os.makedirs(dir_path, exist_ok=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)       
        except Exception as e:
            raise NetworkSecurityException(e, sys)