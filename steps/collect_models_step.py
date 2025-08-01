from typing import Dict
from zenml import step

@step
def collect_models_step(
    svm_eval: Dict[str, float],
    rf_eval: Dict[str, float],
    nb_eval: Dict[str, float],
    xg_eval: Dict[str, float],
    logistic_eval: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Collects evaluation metrics from different models into a single dictionary.

    Returns:
        dict: A dictionary with model names as keys and their metrics as values.
    """
    return {
        "svm": svm_eval,
        "rf": rf_eval,
        "nb": nb_eval,
        "xg": xg_eval,
        "logistic": logistic_eval,
    }
