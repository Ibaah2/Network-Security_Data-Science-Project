import os
import sys
import numpy as np
import pandas as pd

'''
Defining common constant variable for training pipeline
'''

TARGET_COLUMN = 'Result'
PIPELINE_NAME: str = 'NetworkSecurity'
ARTIFACT_DIR: str = 'Artifacts'
FILE_NAME: str = 'NetworkData.csv'

TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'


'''
Data Ingestion Configuration related constant 
'''

DATA_INGESTION_COLLECTION_NAME: str = 'NetworkData'
DATA_INGESTION_DATABASE_NAME: str = 'baah'
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
DATA_INGESTION_INGESTION_DIR: str = 'ingested'
DATA_TRAIN_TEST_SPLIT_RATIO: float = 0.2


'''
Data Validation related constant 
'''

DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_VALID_DIR: str = 'validated'
DATA_VALIDATION_INVALID_DIR: str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR: str = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = 'report.yaml'

SCHEMA_FILE_NAME: str = 'schema.yaml'
SCHEMA_FILE_PATH: str = os.path.join('data_schema', SCHEMA_FILE_NAME)


'''
Training Pipeline Configuration related constant 
'''

DATA_TRANSFORMATION_DIR_NAME: str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = 'transformed_object'
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME: str = 'transformed_train.npy'
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME: str = 'transformed_test.npy'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME: str = 'transformed_object.pkl'

DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    'missing_values': np.nan,
    'n_neighbors': 3,
    'weights': 'uniform'
}
