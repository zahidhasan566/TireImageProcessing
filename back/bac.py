from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from google.cloud import vision

# Google Vision API setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './credential/vital-code-441506-t5-1bc3561300e6.json'
client = vision.ImageAnnotatorClient()

app = FastAPI()

# Pydantic model to receive image path as JSON
class ImagePath(BaseModel):
    path: str

# Main function to handle OCR requests
def image_to_text(image_path: str) -> str:
    # Step 1: Process image and make Vision API call
    image = vision.Image()
    with open(image_path, 'rb') as image_file:
        image.content = image_file.read()

    response = client.text_detection(image=image)

    # Check for errors in the response
    if response.error.message:
        raise HTTPException(status_code=500, detail=f"API Error: {response.error.message}")

    # Extract and return text from response
    detected_text = response.text_annotations[0].description if response.text_annotations else ""

    return detected_text

# FastAPI route to handle image OCR request
@app.post("/extract-text-from-image/")
async def extract_text_from_image(image: ImagePath):
    try:
        print("Received request with image path:", image.path)  # Debug log
        detected_text = image_to_text(image.path)
        return {"detected_text": detected_text}
    except HTTPException as e:
        print("HTTP Exception:", e.detail)  # Debug log
        raise e
    except Exception as e:
        print("Exception:", str(e))  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

# Example: This part is just to show how you can run it in the script (not required in FastAPI app)
if __name__ == "__main__":
    # FastAPI server will run here
    imagepath = './srcImg/p3.jpeg'
    print(image_to_text(imagepath))
    pass