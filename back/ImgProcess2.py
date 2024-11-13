from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from google.cloud import vision
import jwt
from datetime import datetime, timedelta

# Google Vision API setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './credential/vital-code-441506-t5-1bc3561300e6.json'
client = vision.ImageAnnotatorClient()

# Pydantic model to receive image path as JSON
class ImagePath(BaseModel):
    path: str

# Dummy functions for user data
def get_user_quota(user_id):
    # Retrieve user quota from your database
    return 100  # For example, max 100 API calls


def get_user_usage(user_id):
    # Retrieve current usage count from your database
    return 20  # Example usage count


def update_user_usage(user_id, increment=1):
    # Increment the user's usage count in your database
    pass


def is_user_authorized(user_id, token):
    # Decode token and verify user permissions
    try:
        decoded = jwt.decode(token, "your_secret_key", algorithms=["HS256"])
        return decoded["user_id"] == user_id
    except jwt.ExpiredSignatureError:
        return False


# Main function to handle OCR requests
def image_to_text(user_id, token, image_path):
    # Step 1: Authorization check
    if not is_user_authorized(user_id, token):
        return "Unauthorized request."

    # Step 2: Check API usage against user quota
    if get_user_usage(user_id) >= get_user_quota(user_id):
        return "Quota exceeded. Upgrade plan or wait for reset."

    # Step 3: Process image and make Vision API call
    image = vision.Image()
    with open(image_path, 'rb') as image_file:
        image.content = image_file.read()

    response = client.text_detection(image=image)

    # Check for errors in the response
    if response.error.message:
        raise Exception(f"API Error: {response.error.message}")

    # Extract and return text from response
    detected_text = response.text_annotations[0].description if response.text_annotations else ""

    # Step 4: Log the usage
    update_user_usage(user_id)

    return detected_text


# Example call
user_id = 1
user_token = jwt.encode({"user_id": user_id, "exp": datetime.utcnow() + timedelta(hours=1)}, "your_secret_key",
                        algorithm="HS256")
image_path = './srcImg/p3.jpeg'
print(image_to_text(user_id, user_token, image_path))
