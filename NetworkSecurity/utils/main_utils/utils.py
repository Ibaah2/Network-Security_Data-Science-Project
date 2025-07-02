import yaml
import os 
import sys
import pickle
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from tempfile import gettempdir


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
    
def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to a file.
    
    Args:
        file_path (str): Path to the file where the array will be saved.
        array (np.ndarray): NumPy array to save.
    
    Raises:
        NetworkSecurityException: If the file cannot be written.
    """
    try:
        logging.info(f"Saving NumPy array to {file_path}.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(f"Error saving NumPy array to {file_path}: {e}", sys) from e
    

def save_object(file_path: str, obj: object) -> None:
    """
    Saves an object to a file using pickle.
    
    Args:
        file_path (str): Path to the file where the object will be saved.
        obj (object): Object to save.
    
    Raises:
        NetworkSecurityException: If the file cannot be written.
    """
    try:
        logging.info(f"Saving object to {file_path}.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:            
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}.")
    except:
        # Fall to temporary path if existing path does not work
        temp_path = os.path.join(gettempdir(), 'NetworkSecurity', 'Artifacts')
        os.makedirs(temp_path, exist_ok=True)
        return temp_path
    
def load_object(file_path: str) -> object:
    """
    Loads an object from a file using pickle.
    
    Args:
        file_path (str): Path to the file from which the object will be loaded.
    
    Returns:
        object: Loaded object.
    
    Raises:
        NetworkSecurityException: If the file cannot be read or parsed.
    """
    try:
        logging.info(f"Loading object from {file_path}.")
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info("Object loaded successfully.")
        return obj
    except Exception as e:
        raise NetworkSecurityException(f"Error loading object from {file_path}: {e}", sys) from e
    

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a file.
    
    Args:
        file_path (str): Path to the file from which the array will be loaded.
    
    Returns:
        np.ndarray: Loaded NumPy array.
    
    Raises:
        NetworkSecurityException: If the file cannot be read or parsed.
    """
    try:
        logging.info(f"Loading NumPy array from {file_path}.")
        with open(file_path, 'rb') as file_obj:
            array = np.load(file_obj)
        logging.info("NumPy array loaded successfully.")
        return array
    except Exception as e:
        raise NetworkSecurityException(f"Error loading NumPy array from {file_path}: {e}", sys) from e
    

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict)-> dict:
    """
    Evaluates multiple machine learning models and returns the best model based on accuracy.
    
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        models (dict): Dictionary of model names and their instances.
        params (dict): Dictionary of model parameters.
    
    Returns:
        tuple: Best model name and its accuracy score.
    """
    try:
        report = {}
        best_model_name = None
        best_model = None
        best_score = 0
        
        # Iterate through each model and evaluate it
        for model_name, model in models.items():
            if model_name not in params:
                print(f'[WARNING] No hyperparameters provided for {model_name}, skipping.')
                continue

            print(f'[INFO] Evaluating {model_name}')
            param_grid = params[model_name]

            try:
                # Fit the model
                logging.info(f"Training model .....")
                grid_search_cv = GridSearchCV(estimator=model, 
                                param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)            
                grid_search_cv.fit(X_train, y_train)
                best_params = grid_search_cv.best_params_
                logging.info(f'{model_name} best parameters: {best_params}')

                # Get best paramters from each model and set them to the model
                model.set_params(**best_params)
                model.fit(X_train, y_train)
                
                # Predict on the train and test sets
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate r2 score
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                # Store the results in the report
                report[model_name] = test_model_score

                if test_model_score > best_score:
                    best_score = test_model_score
                    best_model_name = model_name
                    best_model = model
                logging.info(f"{model_name} accuracy: {test_model_score:.4f}")
            except Exception as e:
                logging.info(f'Failed to train model {model_name}')
        return best_model_name, best_model, report    
    except Exception as e:
        raise NetworkSecurityException(f"Error evaluating models: {e}", sys) from e