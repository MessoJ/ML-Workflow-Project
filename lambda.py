
"""
preprocessImage

A lambda function that copies an object from S3, base64 encodes it, and 
then return it (serialized data) to the step function as `image_data` in an event.
"""
import json
import boto3
import base64

s3 = boto3.client("s3")

def lambda_handler(event, context):
    """
    Input expected from Step Functions:
      {
        "image_data": "",
        "s3_bucket": "<your-bucket>",
        "s3_key": "test/bicycle_s_000513.png"
      }
    """
    bucket = event["s3_bucket"]
    key    = event["s3_key"]

    obj = s3.get_object(Bucket=bucket, Key=key)
    b64 = base64.b64encode(obj["Body"].read()).decode("utf-8")

    # Pass forward the same shape Step Functions expects (body only is forwarded)
    out = {
        "image_data": b64,
        "s3_bucket": bucket,
        "s3_key": key
    }
    return {"statusCode": 200, "body": out}











"""
classifyImage: Image-Classification

A lambda function that is responsible for the classification part. It takes the image output from the 
lambda 1 function, decodes it, and then pass inferences back to the the Step Function
"""
import os
import json
import base64
import boto3

smr = boto3.client("sagemaker-runtime")
ENDPOINT = os.environ["ENDPOINT"]  # img-classification-job-2025-08-21-21-15-29-564

def lambda_handler(event, context):
    # event already IS the body from the previous step
    # event = { "image_data": "<base64...>", ... }
    b = base64.b64decode(event["image_data"])

    resp = smr.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="image/png",
        Body=b,
        Accept="application/json"
    )

    payload = resp["Body"].read().decode("utf-8").strip()
    try:
        probs = json.loads(payload)          # some containers return JSON
    except json.JSONDecodeError:
        probs = [float(x) for x in payload.strip("[] \n").split(",")]  # some return plain list

    # pass forward the same shape the state machine expects
    event["inferences"] = probs
    return {"statusCode": 200, "body": event}









"""
filterResults:Filter-Low-Confidence-Inferences

A lambda function that takes the inferences from the Lambda 2 function output and filters low-confidence inferences
(above a certain threshold indicating success)
"""
import os
import json

THRESHOLD = float(os.environ.get("THRESHOLD", "0.70"))

def lambda_handler(event, context):
    # event already IS the body from the previous state
    inferences = event["inferences"]       # <-- no ['body']

    if max(inferences) < THRESHOLD:
        # Fail loudly for low confidence
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    # Otherwise pass the same shape forward
    return {"statusCode": 200, "body": event}
