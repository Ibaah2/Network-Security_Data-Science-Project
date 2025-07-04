from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.utils.main_utils.utils import load_object
from NetworkSecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME
from NetworkSecurity.pipeline.training_pipeline import TrainingPipeline
import os
import sys
import numpy as np
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

import pymongo
import certifi 
ca = certifi.where()
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")  
cluster = os.getenv("MONGO_CLUSTER")

uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = pymongo.MongoClient(uri, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods =['*'],
    allow_headers=['*']
)

@app.get('/', tags = ['authentication'])
async def index():
    return RedirectResponse(uri='/docs')

@app.get('/train')
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response('Training Successful')
    except Exception as e:
        raise NetworkSecurityException(e, sys)

if __name__ == '__main__':
    app_run(app, host = 'localhost', port = 8000)