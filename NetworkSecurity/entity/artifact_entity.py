from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to hold the artifacts related to data ingestion.
    """
    training_file_path: str
    testing_file_path: str

@dataclass
class DataValidationArtifact:
    """
    Data Validation Artifact class to hold the artifacts related to data validation.
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str
    
