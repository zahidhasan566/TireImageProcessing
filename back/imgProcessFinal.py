from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import vision
import os
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

# Google Vision API setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './credential/vital-code-441506-t5-1bc3561300e6.json'
client = vision.ImageAnnotatorClient()


# Pydantic model to receive image path as JSON
class ImagePath(BaseModel):
    path: str


def preprocess_image(image_path: str):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found.")

    # Open the image using PIL
    image = Image.open(image_path)

    # Convert image to numpy array (OpenCV format)
    open_cv_image = np.array(image.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply some preprocessing: Gaussian Blur, Binary Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Apply dilation and erosion to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    processed_image = cv2.dilate(binary, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)

    # Convert processed image to bytes for Vision API
    success, buffer = cv2.imencode('.png', processed_image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process image.")

    return buffer.tobytes()


@app.post("/extract-text-from-path/")
async def extract_text_from_path(image_path: ImagePath):
    # Preprocess image based on the path provided in the JSON payload
    processed_image_bytes = preprocess_image(image_path.path)

    # Create Vision API image object
    image = vision.Image(content=processed_image_bytes)

    # Call Google Vision API to extract text
    response = client.text_detection(image=image)

    # Check for errors in the response
    if response.error.message:
        raise HTTPException(status_code=500, detail=f"Google Vision API Error: {response.error.message}")

    # Extract detected text from the API response
    detected_text = response.text_annotations[0].description if response.text_annotations else "No text detected."

    return {"detected_text": detected_text}
