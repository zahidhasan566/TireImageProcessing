from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import base64
from google.cloud import vision
from datetime import datetime
import uuid  # For generating unique IDs (though we'll use a counter)

# Google Vision API setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './credential/vital-code-441506-t5-1bc3561300e6.json'
client = vision.ImageAnnotatorClient()
# In-memory ID counter, initialized to 0
current_id = 0
app = FastAPI()

# Pydantic model to receive image data as base64 in JSON
class ImageData(BaseModel):
    base64_image: str  # The base64 encoded image data

# Main function to handle OCR requests from base64
def image_to_text(base64_image: str) -> str:
    # Step 1: Decode the base64 image data
    try:
        image_content = base64.b64decode(base64_image)
        print("Image decoded successfully.")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    # Step 2: Prepare image for Google Vision API
    image = vision.Image(content=image_content)

    # Step 3: Make the Vision API call
    response = client.text_detection(image=image)

    # Step 4: Check for errors in the response
    if response.error.message:
        raise HTTPException(status_code=500, detail=f"API Error: {response.error.message}")

    # Step 5: Extract and return text from response
    detected_text = response.text_annotations[0].description if response.text_annotations else ""

    # Step 6: Replace newline characters (\n) with spaces
    detected_text = detected_text.replace("\n", " ")
    return detected_text


# Function to append detected text to a text file with ID and DateTime
def append_text_to_file(detected_text: str) -> str:
    global current_id  # Access the global counter

    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate the filename (you could also use the current ID here if needed)
    filename = "detected_texts.txt"
    file_path = os.path.join("output_files", filename)

    # Ensure the output directory exists
    os.makedirs("output_files", exist_ok=True)

    # Append the detected text with ID and DateTime
    with open(file_path, "a") as file:
        # If the file is not empty, append a new line first
        if os.path.getsize(file_path) > 0:
            file.write("\n")  # Line break before the new entry
        file.write(f"ID: {current_id}, DateTime: {current_time}, Detected Text: {detected_text}\n")

    print(f"Detected text appended to {file_path}")

    # Increment the ID for the next request
    current_id += 1

    return file_path

# FastAPI route to handle image OCR request
@app.post("/extract-text-from-base64-image/")
async def extract_text_from_base64_image(image: ImageData):
    try:
        print("Received request with base64 image data.")
        detected_text = image_to_text(image.base64_image)
        # Step 2: Save detected text to a file with auto-incremented ID
        file_path = append_text_to_file(detected_text)
        return {"detected_text": detected_text}
    except HTTPException as e:
        print("HTTP Exception:", e.detail)
        raise e
    except Exception as e:
        print("Exception:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
