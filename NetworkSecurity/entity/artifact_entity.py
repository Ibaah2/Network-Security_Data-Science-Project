from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to hold the artifacts related to data ingestion.
    """
    training_file_path: str
    testing_file_path: str
    
