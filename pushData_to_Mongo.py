from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv
import os
import sys

load_dotenv()  # Loads variables from .env text file

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")


# -----------------------------
# MongoDB Atlas Connection URI
# -----------------------------
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
    """
    This class handles reading network data from a CSV file, converting it to JSON, and inserting it into MongoDB.
    """

    def __init__(self):
        """
        Class constructor. Can be expanded in future to load configurations.
        """
        try:
            logging.info("NetworkDataExtract class initialized.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json(self, file_path):
        """
        Reads data from a CSV file and converts it to a list of JSON records.

        Args:
            file_path (str): Path to the input CSV file.

        Returns:
            list: List of dictionary records ready for MongoDB insertion.
        """
        try:
            logging.info(f"Reading CSV file from: {file_path}")
            data = pd.read_csv(file_path)

            logging.info(f"CSV file successfully read with {len(data)} rows.")
            
            # Reset index to ensure clean conversion
            data.reset_index(drop=True, inplace=True) # Drop index

            logging.info("Converting DataFrame to JSON records.")
            records = list(json.loads(data.T.to_json()).values())

            logging.info(f"Successfully converted data into {len(records)} JSON records.")
            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_MongoDB(self, records, database, collection):
        """
        Inserts a list of records into a specified MongoDB collection.

        Args:
            records (list): List of JSON-like dicts to be inserted.
            database (str): Name of the MongoDB database.
            collection (str): Name of the collection to insert into.

        Returns:
            int: Number of records successfully inserted.
        """
        try:
            logging.info("Establishing connection to MongoDB Atlas...")
            mongo_client = pymongo.MongoClient(uri)

            # Access or create the specified database and collection
            db = mongo_client[database]
            coll = db[collection]

            logging.info(f"Inserting {len(records)} records into {database}.{collection} collection...")
            result = coll.insert_many(records)

            logging.info(f"{len(result.inserted_ids)} records successfully inserted.")
            return len(result.inserted_ids)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


# ------------------------------
# Main Program Execution Starts
# ------------------------------
if __name__ == "__main__":
    try:
        # Path to the CSV file containing the phishing network data
        file_path = 'NetworkData/phisingData.csv'

        # MongoDB Atlas database and collection names
        database = 'baah'
        collection = 'NetworkData'

        logging.info("Starting the network data extraction and insertion process...")

        # Create an instance of the data extraction class
        networkobj = NetworkDataExtract()

        # Convert CSV data to JSON format
        records = networkobj.csv_to_json(file_path=file_path)

        # Insert the converted records into MongoDB
        no_of_records = networkobj.insert_data_MongoDB(
            records=records,
            database=database,
            collection=collection
        )

        # Print and log the final result
        logging.info(f"Total records inserted into MongoDB: {no_of_records}")
        print(f"Total records inserted: {no_of_records}")

    except Exception as e:
        # Catch and log any unexpected errors during execution
        logging.error(f"An error occurred in the main execution block: {e}")
        raise NetworkSecurityException(e, sys)

