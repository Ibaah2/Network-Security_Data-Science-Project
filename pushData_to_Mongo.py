from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv
import os
import sys

load_dotenv()  # Loads variables from .env text file

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")

uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Trying out the mongoDB to see if we are able to connect successfully.
# Send a ping to confirm a successful connection
#try:
 #   client.admin.command('ping')
  #  print("Pinged your deployment. You successfully connected to MongoDB!")
#except Exception as e:
 #   print(e)

# It was a success at the end


#################################
    # Extracting the data
#################################

import certifi
ca = certifi.where()

# Import libraries
import json
import pandas as pd
import numpy as np
import pymongo 
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.exception_handling.exception import NetworkSecurityException

class NetworkDataExtract:

    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json(self, file_path):
        try:
            logging.info('Reading csv data')
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)

            logging.info('Converting csv data to json format')
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_MongoDB(self, records, database, collection):
        try:
            mongo_client = pymongo.MongoClient(uri)
            db = mongo_client[database]
            coll = db[collection]
            coll.insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    file_path = 'NetworkData/phisingData.csv'
    database = 'baah'
    collection = 'NetworkData'

    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json(file_path=file_path)
    no_of_records = networkobj.insert_data_MongoDB(records=records, database=database, collection=collection)

    print(no_of_records)
