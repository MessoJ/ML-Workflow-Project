# ================================
# Lambda 1: serializeImageData.py
# ================================
def serializeImageData_handler(event, context):
    bucket = event["s3_bucket"]
    key = event["s3_key"]

    return {
        "s3_bucket": bucket,
        "s3_key": key
    }


# ================================
# Lambda 2: imageClassifier.py
# ================================
import boto3

s3 = boto3.client("s3")

def imageClassifier_handler(event, context):
    bucket = event["s3_bucket"]
    key = event["s3_key"]

    # Fetch image (not used, but simulates real classifier reading bytes)
    _ = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    # Fake inference output
    event["inferences"] = [0.1, 0.9]

    return event


# ================================
# Lambda 3: inferenceConfidenceFilter.py
# ================================
THRESHOLD = 0.7

def inferenceConfidenceFilter_handler(event, context):
    inferences = event.get("inferences", [])
    
    if max(inferences) >= THRESHOLD:
        return event
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
