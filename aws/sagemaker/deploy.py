"""
AWS SageMaker Deployment Script for Railway Sign Language Translation
Handles model deployment, endpoint creation, and inference configuration
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import json
import os
from datetime import datetime


class SageMakerDeployment:
    """Handle SageMaker model deployment and endpoint management"""
    
    def __init__(self, region='us-east-1'):
        """
        Initialize SageMaker deployment
        
        Args:
            region: AWS region for deployment
        """
        self.region = region
        self.session = sagemaker.Session()
        self.bucket = self.session.default_bucket()
        
        try:
            self.role = get_execution_role()
        except:
            # If not running in SageMaker, use IAM role ARN
            self.role = os.environ.get('SAGEMAKER_ROLE_ARN')
            if not self.role:
                raise ValueError("SAGEMAKER_ROLE_ARN environment variable must be set")
        
        self.sm_client = boto3.client('sagemaker', region_name=region)
    
    
    def upload_model_artifacts(self, model_path, model_name='railway-sign-model'):
        """
        Upload model artifacts to S3
        
        Args:
            model_path: local path to model.tar.gz
            model_name: name for the model in S3
            
        Returns:
            S3 URI of uploaded model
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        s3_key = f'models/{model_name}/{timestamp}/model.tar.gz'
        s3_uri = f's3://{self.bucket}/{s3_key}'
        
        print(f"Uploading model to {s3_uri}...")
        
        # Upload to S3
        s3_client = boto3.client('s3', region_name=self.region)
        s3_client.upload_file(model_path, self.bucket, s3_key)
        
        print(f"Model uploaded successfully")
        return s3_uri
    
    
    def create_pytorch_model(self, model_s3_uri, framework_version='2.0'):
        """
        Create PyTorch model object for deployment
        
        Args:
            model_s3_uri: S3 location of model artifacts
            framework_version: PyTorch version
            
        Returns:
            PyTorchModel object
        """
        pytorch_model = PyTorchModel(
            model_data=model_s3_uri,
            role=self.role,
            framework_version=framework_version,
            py_version='py310',
            entry_point='inference.py',
            source_dir='./inference',
            name=f'railway-sign-model-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        
        return pytorch_model
    
    
    def deploy_endpoint(self, 
                       model_s3_uri,
                       endpoint_name='railway-sign-translator',
                       instance_type='ml.m5.xlarge',
                       instance_count=1):
        """
        Deploy model to SageMaker endpoint
        
        Args:
            model_s3_uri: S3 location of model
            endpoint_name: name for the endpoint
            instance_type: EC2 instance type
            instance_count: number of instances
            
        Returns:
            Endpoint name
        """
        print(f"Creating endpoint: {endpoint_name}")
        
        # Create model object
        pytorch_model = self.create_pytorch_model(model_s3_uri)
        
        # Deploy model
        predictor = pytorch_model.deploy(
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            initial_instance_count=instance_count,
            wait=True
        )
        
        print(f"Endpoint {endpoint_name} deployed successfully")
        return endpoint_name
    
    
    def create_autoscaling(self, 
                          endpoint_name,
                          variant_name='AllTraffic',
                          min_capacity=1,
                          max_capacity=5,
                          target_value=70.0):
        """
        Configure autoscaling for endpoint
        
        Args:
            endpoint_name: name of endpoint
            variant_name: production variant name
            min_capacity: minimum number of instances
            max_capacity: maximum number of instances
            target_value: target invocations per instance
        """
        client = boto3.client('application-autoscaling', region_name=self.region)
        
        resource_id = f'endpoint/{endpoint_name}/variant/{variant_name}'
        
        # Register scalable target
        client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity
        )
        
        # Create scaling policy
        client.put_scaling_policy(
            PolicyName=f'{endpoint_name}-scaling-policy',
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': target_value,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleInCooldown': 300,
                'ScaleOutCooldown': 60
            }
        )
        
        print(f"Autoscaling configured for {endpoint_name}")
    
    
    def invoke_endpoint(self, endpoint_name, video_data):
        """
        Invoke SageMaker endpoint for inference
        
        Args:
            endpoint_name: name of endpoint
            video_data: video data (base64 or frames)
            
        Returns:
            Prediction results
        """
        runtime = boto3.client('sagemaker-runtime', region_name=self.region)
        
        payload = json.dumps({
            'video_data': video_data
        })
        
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        result = json.loads(response['Body'].read().decode())
        return result
    
    
    def delete_endpoint(self, endpoint_name):
        """
        Delete SageMaker endpoint
        
        Args:
            endpoint_name: name of endpoint to delete
        """
        print(f"Deleting endpoint: {endpoint_name}")
        
        self.sm_client.delete_endpoint(EndpointName=endpoint_name)
        
        # Also delete endpoint configuration
        try:
            self.sm_client.delete_endpoint_config(
                EndpointConfigName=endpoint_name
            )
        except:
            pass
        
        print(f"Endpoint {endpoint_name} deleted")
    
    
    def list_endpoints(self):
        """
        List all active endpoints
        
        Returns:
            List of endpoint names
        """
        response = self.sm_client.list_endpoints(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=100
        )
        
        endpoints = [ep['EndpointName'] for ep in response['Endpoints']]
        return endpoints


def main():
    """Example deployment workflow"""
    
    # Initialize deployment
    deployer = SageMakerDeployment(region='us-east-1')
    
    # Upload model
    model_path = './models/railway_sign_model.tar.gz'
    model_s3_uri = deployer.upload_model_artifacts(model_path)
    
    # Deploy endpoint
    endpoint_name = 'railway-sign-translator-prod'
    deployer.deploy_endpoint(
        model_s3_uri=model_s3_uri,
        endpoint_name=endpoint_name,
        instance_type='ml.m5.xlarge',
        instance_count=1
    )
    
    # Configure autoscaling
    deployer.create_autoscaling(
        endpoint_name=endpoint_name,
        min_capacity=1,
        max_capacity=5,
        target_value=70.0
    )
    
    print(f"\nDeployment complete!")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Region: {deployer.region}")


if __name__ == '__main__':
    main()