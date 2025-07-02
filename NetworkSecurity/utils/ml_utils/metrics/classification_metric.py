from NetworkSecurity.entity.artifact_entity import ClassificationMetricArtifact
from NetworkSecurity.exception_handling.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics and return them as a ClassificationMetricArtifact.
    
    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
    
    Returns:
        ClassificationMetricArtifact: Artifact containing classification metrics.
    """
    try:
        logging.info("Calculating classification metrics.")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')

        return ClassificationMetricArtifact(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)