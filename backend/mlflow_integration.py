"""
MLflow Integration for Model Versioning and Experiment Tracking

This module provides utilities for:
- Logging model training metrics and parameters
- Versioning model artifacts
- Tracking experiment runs
- Serving models via MLflow
"""

from datetime import datetime
import logging
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker for recommendation models.

    Tracks:
    - Training metrics (HR@10, NDCG@10, loss)
    - Model parameters (learning rate, batch size, etc.)
    - Model artifacts (PyTorch models, TurboVec indices)
    - Ensemble weights and configuration
    """

    def __init__(
        self,
        experiment_name: str = "apex-recommendation-engine",
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local file://)
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set up MLflow experiment: {e}")
            raise

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Dictionary of tags for the run

        Returns:
            Active MLflow run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name, tags=tags or {})
        logger.info(f"Started MLflow run: {run_name}")
        return run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log training parameters."""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log training metrics."""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics at step {step}")

    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "pytorch",
        input_example: Any | None = None,
        signature: Any | None = None,
    ) -> None:
        """
        Log model artifact.

        Args:
            model: Model object to log
            model_name: Name for the model
            model_type: Type of model (pytorch, sklearn, etc.)
            input_example: Example input for the model
            signature: Model signature
        """
        if model_type == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path=model_name, input_example=input_example)
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path=model_name, input_example=input_example)
        else:
            # Fallback to generic artifact logging
            mlflow.log_artifact(model, artifact_path=model_name)

        logger.info(f"Logged model: {model_name}")

    def log_ensemble_weights(self, weights: dict[str, float]) -> None:
        """Log ensemble model weights."""
        mlflow.log_dict(weights, "ensemble_weights.json")
        logger.info(f"Logged ensemble weights: {weights}")

    def log_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        """Log arbitrary artifact file."""
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
        logger.info(f"Logged artifact: {file_path}")

    def log_figure(self, figure, artifact_file: str) -> None:
        """Log matplotlib/plotly figure."""
        mlflow.log_figure(figure, artifact_file)
        logger.info(f"Logged figure: {artifact_file}")

    def log_dataset_info(self, dataset_info: dict[str, Any]) -> None:
        """Log dataset information."""
        mlflow.log_dict(dataset_info, "dataset_info.json")
        logger.info("Logged dataset info")

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")

    def get_best_model(self, metric_name: str, order: str = "max") -> dict[str, Any] | None:
        """
        Get the best model run based on a metric.

        Args:
            metric_name: Name of the metric to optimize
            order: "max" or "min" for optimization direction

        Returns:
            Dictionary with best run information
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if order == 'max' else 'ASC'}"],
        )

        if len(runs) == 0:
            logger.warning(f"No runs found for experiment {self.experiment_name}")
            return None

        best_run = runs.iloc[0]
        logger.info(f"Best run for {metric_name}: {best_run['run_id']}")

        return {
            "run_id": best_run["run_id"],
            "metrics": best_run["metrics"].to_dict(),
            "params": best_run["params"].to_dict(),
            "artifact_uri": best_run["artifact_uri"],
        }

    def load_model(self, run_id: str, model_name: str) -> Any:
        """
        Load a model from a specific run.

        Args:
            run_id: MLflow run ID
            model_name: Name of the model artifact

        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        return model

    def compare_runs(self, run_ids: list[str]) -> pd.DataFrame:
        """
        Compare multiple runs side by side.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with comparison results
        """
        import pandas as pd

        runs_data = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            runs_data.append({"run_id": run_id, **run.data.metrics, **run.data.params})

        return pd.DataFrame(runs_data)


class ModelRegistry:
    """
    MLflow Model Registry for production model management.

    Manages:
    - Model versioning
    - Model staging (Staging, Production, Archived)
    - Model deployment
    """

    def __init__(self, model_name: str = "apex-ensemble"):
        """
        Initialize model registry.

        Args:
            model_name: Name of the registered model
        """
        self.model_name = model_name
        self._ensure_registered_model()

    def _ensure_registered_model(self) -> None:
        """Ensure the model is registered in MLflow."""
        try:
            mlflow.get_registered_model(self.model_name)
        except Exception:
            mlflow.create_registered_model(self.model_name)
            logger.info(f"Created registered model: {self.model_name}")

    def register_model(
        self, run_id: str, model_name: str, stage: str = "Staging", description: str | None = None
    ) -> str:
        """
        Register a model from a run.

        Args:
            run_id: MLflow run ID
            model_name: Name of the model artifact in the run
            stage: Stage to register the model in
            description: Description of the model version

        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{model_name}"

        model_version = mlflow.register_model(model_uri=model_uri, name=self.model_name, description=description)

        # Transition to specified stage
        mlflow.transition_model_version_stage(name=self.model_name, version=model_version.version, stage=stage)

        logger.info(f"Registered model version {model_version.version} in stage {stage}")
        return model_version.version

    def get_production_model(self) -> Any | None:
        """Get the current production model."""
        try:
            model_uri = f"models:/{self.model_name}/Production"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded production model from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None

    def promote_to_production(self, version: str) -> None:
        """Promote a model version to production."""
        mlflow.transition_model_version_stage(name=self.model_name, version=version, stage="Production")
        logger.info(f"Promoted model version {version} to Production")

    def archive_model(self, version: str) -> None:
        """Archive a model version."""
        mlflow.transition_model_version_stage(name=self.model_name, version=version, stage="Archived")
        logger.info(f"Archived model version {version}")


# Convenience functions for quick usage
def get_tracker(experiment_name: str = "apex-recommendation-engine", tracking_uri: str | None = None) -> MLflowTracker:
    """Get an MLflow tracker instance."""
    return MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)


def get_registry(model_name: str = "apex-ensemble") -> ModelRegistry:
    """Get a model registry instance."""
    return ModelRegistry(model_name=model_name)
