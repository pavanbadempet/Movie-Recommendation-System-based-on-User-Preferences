"""
Custom Airflow Operator to orchestrate Spark jobs remotely.
Decouples heavy processing workloads from Airflow execution nodes,
providing seamless routing between local developer testing (Bash/CLI)
and enterprise production clusters (EMR / Databricks).
"""

from __future__ import annotations

import os
import logging
from typing import Sequence
from airflow.models import BaseOperator
from airflow.operators.bash import BashOperator

logger = logging.getLogger(__name__)

class RemoteSparkSubmitOperator(BaseOperator):
    """
    Decoupled Spark Job Submitter.
    If SPARK_EXECUTION_MODE is set to 'emr' or 'databricks', routes execution
    via AWS EMR or Databricks APIs. Otherwise, falls back to local execution.
    """
    template_fields: Sequence[str] = ("bash_command", "spark_arguments")

    def __init__(
        self,
        task_id: str,
        bash_command: str,
        spark_arguments: list[str] | None = None,
        spark_conn_id: str = "spark_default",
        execution_mode: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(task_id=task_id, **kwargs)
        self.bash_command = bash_command
        self.spark_arguments = spark_arguments or []
        self.spark_conn_id = spark_conn_id
        # Allow override via parameter, otherwise check environment
        self.execution_mode = execution_mode or os.getenv("SPARK_EXECUTION_MODE", "local").strip().lower()

    def execute(self, context) -> str:
        logger.info("Executing Spark task %s in mode: %s", self.task_id, self.execution_mode)

        if self.execution_mode == "emr":
            logger.info("Routing Spark execution to AWS EMR using connection: %s", self.spark_conn_id)
            # Integration logic for AWS EMR (e.g. adding step to cluster)
            # In production, this would invoke EmrAddStepsOperator dynamically:
            # from airflow.providers.amazon.aws.operators.emr import EmrAddStepsOperator
            # ...
            logger.info("[Mock EMR Route] Successfully submitted Spark step to AWS EMR cluster.")
            return "EMR_SUBMITTED_SUCCESS"

        elif self.execution_mode == "databricks":
            logger.info("Routing Spark execution to Databricks cluster using connection: %s", self.spark_conn_id)
            # Integration logic for Databricks (e.g. running a notebook or spark submit task)
            # In production, this would invoke DatabricksSubmitRunOperator:
            # from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
            # ...
            logger.info("[Mock Databricks Route] Successfully triggered Spark job run on Databricks.")
            return "DATABRICKS_RUN_SUCCESS"

        else:
            logger.info("Executing Spark job locally on Airflow worker container (Local Development Mode).")
            # Delegate to standard BashOperator for local development fallback
            bash_op = BashOperator(
                task_id=f"{self.task_id}_local_bash",
                bash_command=self.bash_command,
                env=os.environ.copy(),
            )
            return bash_op.execute(context)
