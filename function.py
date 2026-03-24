import boto3
import json

runtime_client = boto3.client('sagemaker-runtime')

# Make sure your endpoint name is still here!
ENDPOINT_NAME = "YOUR-XGBOOST-ENDPOINT-NAME" 

def lambda_handler(event, context):
    try:
        # Check if the request came from the web (Postman/API Gateway)
        if 'body' in event:
            body = json.loads(event['body'])
            features = body.get('features')
        # Fallback in case you still want to use the AWS Test tab
        else:
            features = event.get('features')
            
        if not features:
            return {
                "statusCode": 400, 
                "body": json.dumps({"error": "No features provided in the event"})
            }

        # Convert to CSV string for XGBoost
        csv_payload = ",".join(str(x) for x in features)

        # Send to SageMaker
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_payload
        )
        
        # Decode the result
        result = response['Body'].read().decode('utf-8')
        prediction_probability = float(result)
        predicted_class = 1 if prediction_probability > 0.5 else 0

        # Return standard HTTP response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'probability': prediction_probability,
                'predicted_class': predicted_class
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
