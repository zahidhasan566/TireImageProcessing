from PIL import Image
import pytesseract
import cv2
import numpy as np
import easyocr




#set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#open cv used for better accuracy
def cvImage(image):
    if image is None:
        print("Error: Image not found or unable to load.")
    else:
        # 1. Resize image to double its size
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # 2. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 3. Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 4. Apply Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # 5. Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(thresh, -1, kernel)

        # 6. Apply Morphological Transformation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)

        # 7. OCR with Tesseract
        config = '--oem 3 --psm 11'  # Adjust the configuration if needed
        text = pytesseract.image_to_string(morph, config=config)

        # Optional: Save processed image for inspection
        cv2.imwrite("processed_tire_image.png", thresh)

        return text


def ocr_with_easyocr(image_path):
    """Extract text from an image using EasyOCR."""
    reader = easyocr.Reader(['en'])  # Supports multiple languages
    result = reader.readtext(image_path)

    text = ""
    for detection in result:
        text += detection[1] + "\n"

    return text

def extract_text_from_image(image_path):
    # Open an image file
    img = Image.open(image_path)
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(img)
    return text


def preprocess_image(image_path):
    """Advanced preprocessing to improve OCR accuracy."""
    # Load the image in color
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adjust the contrast using histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Apply adaptive thresholding to increase text visibility
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Deskewing the image if the text is tilted
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # Rotate the image to correct orientation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def ocr_with_tesseract(image_path):
    """Extract text from an image using Tesseract OCR with advanced config."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Perform OCR using Tesseract with custom configurations
    custom_config = r'--oem 3 --psm 6'  # OEM=3 (best OCR engine), PSM=6 (block of text)
    text = pytesseract.image_to_string(processed_image, config=custom_config)

    return text

# Example usage
image_path = './srcImg/p6.jpeg' #  image path
# Load the image
#image = cv2.imread(image_path)
#customizedImg = cvImage(image)
#text = extract_text_from_image(customizedImg)
#print(customizedImg)

extracted_text = ocr_with_easyocr(image_path)
#extracted_text = ocr_with_tesseract(image_path)
print("Extracted Text:\n", extracted_text)

