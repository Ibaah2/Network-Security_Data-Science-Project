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
