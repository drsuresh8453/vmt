"""
track_experiment.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

MLflow experiment tracking — logs all model runs to S3.
"""

import mlflow
import mlflow.sklearn
import boto3
import joblib
import io
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'vehicle-mileage-project')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')
MLFLOW_URI     = f's3://{BUCKET}/mlflow'

def setup_mlflow():
    os.environ['AWS_ACCESS_KEY_ID']     = AWS_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_KEY
    os.environ['AWS_DEFAULT_REGION']    = REGION
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment('vehicle-mileage-prediction')
    print(f"MLflow tracking URI: {MLFLOW_URI}")

def log_run(model_name, model_obj, params, metrics, X_train, tags=None):
    """Log a single model run to MLflow."""
    setup_mlflow()
    with mlflow.start_run(
        run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):

        # Log params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # Log tags
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        mlflow.set_tag('model_type', model_name)
        mlflow.set_tag('author',     'Suresh D R')
        mlflow.set_tag('project',    'vehicle-mileage-prediction')

        # Log model
        mlflow.sklearn.log_model(model_obj, 'model')

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: {run_id}")
        return run_id

def get_best_run():
    """Get the best model run by MAPE."""
    setup_mlflow()
    runs = mlflow.search_runs(
        experiment_names=['vehicle-mileage-prediction'],
        order_by=['metrics.mape ASC']
    )
    if len(runs) == 0:
        print("No runs found")
        return None

    best = runs.iloc[0]
    print(f"\nBest Run:")
    print(f"  Run ID    : {best['run_id']}")
    print(f"  Model     : {best.get('tags.model_type', 'N/A')}")
    print(f"  MAPE      : {best.get('metrics.mape', 'N/A')}")
    print(f"  MAE       : {best.get('metrics.mae', 'N/A')}")
    print(f"  R2        : {best.get('metrics.r2', 'N/A')}")
    return best

def promote_best_model():
    """Load best model from MLflow and save to S3 production path."""
    best = get_best_run()
    if best is None:
        return

    best_model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")

    # Save to S3
    s3  = boto3.client('s3', region_name=REGION,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY)
    buf = io.BytesIO()
    joblib.dump(best_model, buf)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key='models/best_model.pkl', Body=buf.getvalue())
    print(f"\nBest model promoted to: s3://{BUCKET}/models/best_model.pkl")

    # Save metrics
    metrics = {
        'model'     : best.get('tags.model_type', 'Unknown'),
        'mape'      : best.get('metrics.mape', 0),
        'mae'       : best.get('metrics.mae', 0),
        'r2'        : best.get('metrics.r2', 0),
        'run_id'    : best['run_id'],
        'promoted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    s3.put_object(
        Bucket=BUCKET,
        Key='models/model_metrics.json',
        Body=json.dumps(metrics, indent=2)
    )
    print("Metrics saved to S3")

if __name__ == '__main__':
    print("MLflow Experiment Tracker")
    print("=" * 40)
    get_best_run()
