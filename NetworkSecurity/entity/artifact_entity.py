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

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str   

@dataclass
class ClassificationMetricArtifact:
    """
    Classification Metric Artifact class to hold the metrics related to classification.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    #confusion_matrix: dict

@dataclass
class ModelTrainerArtifact:
    """
    Model Trainer Artifact class to hold the artifacts related to model training.
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
    best_model_name: str
   
