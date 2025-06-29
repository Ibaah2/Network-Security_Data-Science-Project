import yaml
import os 
import sys
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
import numpy as np


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    
    Args:
        file_path (str): Path to the YAML file.
    
    Returns:
        dict: Content of the YAML file.
    
    Raises:
        NetworkSecurityException: If the file cannot be read or parsed.
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            content = yaml.safe_load(yaml_file)
        return content
    except Exception as e:
        raise NetworkSecurityException(f"Error reading YAML file at {file_path}: {e}", sys) from e
    

def write_yaml_file(file_path: str, content: object, replace : bool = False) -> None:
    """
    Writes a dictionary to a YAML file.
    
    Args:
        file_path (str): Path to the YAML file.
        content (dict): Content to write to the YAML file.
    
    Raises:
        NetworkSecurityException: If the file cannot be written.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logging.info(f"Writing content to YAML file at {file_path}.")
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)
    except Exception as e:
        raise NetworkSecurityException(f"Error writing YAML file at {file_path}: {e}", sys) from e