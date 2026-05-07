"""
pipeline/sagemaker_pipeline.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

Creates and manages the SageMaker Pipeline for automated retraining.
Run this ONCE from SageMaker notebook to set up the pipeline.
After setup it runs automatically when triggered by Lambda/EventBridge.
"""

import boto3
import sagemaker
import os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline_context import PipelineSession
from datetime import datetime

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'vehicle-mileage-project')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')
GITHUB_REPO    = os.getenv('GITHUB_REPO',
    'https://github.com/YOUR_USERNAME/vehicle-mileage-mlops.git')
PIPELINE_NAME  = 'vehicle-mileage-pipeline'


def create_pipeline():
    """Create SageMaker Pipeline for automated retraining."""
    print("Creating SageMaker Pipeline...")
    print(f"  Pipeline: {PIPELINE_NAME}")
    print(f"  Bucket  : {BUCKET}")
    print(f"  GitHub  : {GITHUB_REPO}")

    session = sagemaker.Session()
    role    = sagemaker.get_execution_role()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    # Pulls latest code from GitHub + data from S3
    # Runs preprocess.py on the current data
    processor = SKLearnProcessor(
        framework_version='1.0-1',
        role=role,
        instance_type='ml.m5.large',
        instance_count=1,
        env={
            'AWS_ACCESS_KEY_ID'    : AWS_ACCESS_KEY,
            'AWS_SECRET_ACCESS_KEY': AWS_SECRET_KEY,
            'S3_BUCKET'            : BUCKET,
            'AWS_REGION'           : REGION,
        }
    )

    processing_step = ProcessingStep(
        name='PreprocessData',
        processor=processor,
        code='src/preprocess.py',
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=f's3://{BUCKET}/data/current/',
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name='processed',
                source='/opt/ml/processing/output',
                destination=f's3://{BUCKET}/data/processed/'
            )
        ]
    )

    # ── Step 2: Training ──────────────────────────────────────────────────
    # Pulls latest train.py from GitHub
    # Trains on processed data from Step 1
    # Saves best model to S3
    estimator = SKLearn(
        entry_point='src/train.py',
        framework_version='1.0-1',
        instance_type='ml.m5.large',
        role=role,
        output_path=f's3://{BUCKET}/models/sagemaker-output/',
        hyperparameters={
            'n-estimators': 300,
            'max-depth'   : 6,
        },
        environment={
            'AWS_ACCESS_KEY_ID'    : AWS_ACCESS_KEY,
            'AWS_SECRET_ACCESS_KEY': AWS_SECRET_KEY,
            'S3_BUCKET'            : BUCKET,
            'AWS_REGION'           : REGION,
        }
    )

    training_step = TrainingStep(
        name='TrainModel',
        estimator=estimator,
        inputs={
            'train': sagemaker.inputs.TrainingInput(
                s3_data=f's3://{BUCKET}/data/06_encoded_tree.csv',
                content_type='text/csv'
            )
        }
    )

    # ── Build Pipeline ────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        steps=[processing_step, training_step]
    )

    pipeline.upsert(role_arn=role)

    print(f"\nSageMaker Pipeline created: {PIPELINE_NAME}")
    print("Pipeline will run automatically when triggered by:")
    print("  1. Lambda function (daily drift check)")
    print("  2. EventBridge schedule")
    print("  3. Manual trigger from AWS Console")

    return pipeline


def trigger_pipeline(reason="manual"):
    """Manually trigger the pipeline."""
    sm = boto3.client('sagemaker', region_name=REGION,
                       aws_access_key_id=AWS_ACCESS_KEY,
                       aws_secret_access_key=AWS_SECRET_KEY)

    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=f"{reason}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )

    arn = response['PipelineExecutionArn']
    print(f"Pipeline triggered: {arn}")
    return arn


if __name__ == '__main__':
    create_pipeline()
