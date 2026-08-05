"""
Computer Vision & Text Analysis Pipelines
-----------------------------------------
1. Optical Character Recognition (OCR) using pytesseract
2. Object & Bounding Box Detection using OpenCV DNN + MobileNet-SSD
"""

import cv2
import numpy as np
import pytesseract


# =====================================================================
# Pipeline 1: Optical Character Recognition (OCR) via pytesseract
# =====================================================================

def extract_text_from_image(image_path: str, psm_mode: int = 3) -> str:
    """
    Applies image preprocessing (Grayscale, Blur, Adaptive Thresholding) 
    and extracts text using pytesseract with configurable PSM.
    
    Page Segmentation Modes (PSM):
      - 3: Fully automatic page segmentation (Default)
      - 6: Assume a single uniform block of text
      - 7: Treat the image as a single text line
      - 11: Sparse text / find as much text as possible
    """
    # 1. Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 2. Pre-processing steps
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )

    # 3. OCR Engine Configuration
    # --oem 3: Default OCR Engine Mode (LSTM)
    # --psm N: Page Segmentation Mode
    custom_config = f'--oem 3 --psm {psm_mode}'

    # 4. Extract formatted text string
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config)
    return extracted_text


# =====================================================================
# Pipeline 2: Object Detection via cv2.dnn & MobileNet-SSD
# =====================================================================

def detect_objects_mobilenet(
    image_path: str, 
    prototxt_path: str, 
    model_path: str, 
    confidence_threshold: float = 0.5
):
    """
    Pre-processes an image into a 4D blob, feeds it to a MobileNet-SSD network,
    and returns detected bounding box coordinates in (X, Y, W, H) format.
    """
    # 1. Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    (h, w) = image.shape[:2]

    # 2. Load Caffe model network
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # 3. Pre-processing: 4D Blob Construction (blobFromImage)
    # Scale factor = 0.007843 (1/127.5), Target size = 300x300, Mean subtraction = 127.5
    blob = cv2.dnn.blobFromImage(
        image, 
        scalefactor=0.007843, 
        size=(300, 300), 
        mean=127.5
    )

    # 4. Forward pass through DNN
    net.setInput(blob)
    detections = net.forward()

    bounding_boxes = []

    # 5. Extract bounding boxes (X, Y, W, H)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])

            # Scale normalized coordinates back to full image dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Convert (start_x, start_y, end_x, end_y) -> (X, Y, W, H)
            x = start_x
            y = start_y
            width = end_x - start_x
            height = end_y - start_y

            bounding_boxes.append({
                "class_id": class_id,
                "confidence": float(confidence),
                "box": (x, y, width, height)
            })

            # Draw green bounding box on output image
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return bounding_boxes, image


# =====================================================================
# Main Execution Block
# =====================================================================

if __name__ == "__main__":
    # Note: If Tesseract is not added to your system PATH, set the executable path explicitly:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    print("--- 1. Testing OCR Pipeline ---")
    ocr_image_path = "sample_document.png"
    try:
        text_output = extract_text_from_image(ocr_image_path, psm_mode=3)
        print("Extracted Text Output:\n")
        print(text_output)
    except FileNotFoundError as e:
        print(f"[OCR Demo] {e}")

    print("\n--- 2. Testing MobileNet-SSD Detection Pipeline ---")
    ssd_image_path = "sample_objects.jpg"
    prototxt = "MobileNetSSD_deploy.prototxt"
    caffemodel = "MobileNetSSD_deploy.caffemodel"

    try:
        boxes, processed_img = detect_objects_mobilenet(
            image_path=ssd_image_path,
            prototxt_path=prototxt,
            model_path=caffemodel,
            confidence_threshold=0.5
        )

        print("\nDetected Bounding Boxes (X, Y, W, H):")
        for idx, item in enumerate(boxes, 1):
            print(f"Object {idx}: Class {item['class_id']} | Conf: {item['confidence']:.2f} | Box: {item['box']}")

        # Save or display output image with bounding boxes
        cv2.imwrite("detected_objects_output.jpg", processed_img)
        print("\nProcessed image saved to 'detected_objects_output.jpg'")

    except FileNotFoundError as e:
        print(f"[MobileNet-SSD Demo] {e}")
