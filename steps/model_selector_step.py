# model_selector_step.py

from zenml import step
from typing import Dict, Tuple, Any

@step
def model_selector_step(
    model_results: Dict[str, Dict[str, float]],
    primary_metric: str = "F1 Score"
) -> Dict[str, float]:
    """
    Selects the best model based on the primary metric (e.g., F1 Score).
    Returns:
        - Name of the best model
        - Best model object
        - Evaluation metrics dict in original format
    """
    best_model_name = None
    best_metrics = {}
    best_score = float("-inf")

    for model_name, metrics in model_results.items():
        score = metrics.get(primary_metric)
        print(f"{model_name}: {primary_metric} = {score}")
        if score is not None and score > best_score:
            best_model_name = model_name
            best_metrics = metrics
            best_score = score

    # Log best model info to MLflow
    # import mlflow
    # mlflow.set_tag("best_model", best_model_name)
    # mlflow.log_metric(f"best_{primary_metric}", best_score)

    # print(f"\n✅ Best Model: {best_model_name} ({primary_metric}: {best_score:.4f})")

    # Return model name, model object, and full metrics dict
    return best_metrics
