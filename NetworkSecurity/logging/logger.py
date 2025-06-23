import logging
import os
from datetime import datetime

# Creating a unique log file name for any specific time we are logging
# This creates a timestamped filename, like: 06_21_2025_14_30_55.log

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Setting up a directory folder to store the logs
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok = True)

# Create a directory within 'logs' to receive the 'LOG_FILE' with the unique names 
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Setting up logging 
logging.basicConfig(
    filename = LOG_FILE_PATH, #Where loggin files 'LOG_FILE' will be saved
    format = '[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level = logging.INFO
)