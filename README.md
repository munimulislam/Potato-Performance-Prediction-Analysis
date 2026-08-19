Potato Performance Prediction

install dependency:
pip install -e .[data_pipeline,dev]

DATA PIPELINE:

data load:
python -m src.data_load.data_loader

data build:
cd src/data_build
dbt build

data_export:
python -m src.data_export.exporter oa_mart

duckdb ui command:
duckdb -ui

ML EXPERIMENTS:
run scripts from src.ml.experiments.oa.\*

MLFlow server:
mlflow server
