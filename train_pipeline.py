from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    description="Milestone 3 ML pipeline",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="python preprocess.py"
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python train.py"
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="python register_model.py"
    )

    preprocess_data >> train_model >> register_model
