# Task-4-meerabfatima
repository for 4th task

# Computer Vision & Text Analysis: OCR & Object Detection

A Python-based computer vision project implementing two distinct processing pipelines: **Optical Character Recognition (OCR)** using `pytesseract` and **Object Detection & Bounding Box Extraction** using OpenCV's DNN module with a pre-trained **MobileNet-SSD** model.

---

## 📌 Project Overview

This project demonstrates two fundamental approaches in computer vision and text analysis:

1. **Text Extraction Pipeline:** Pre-processes image data using Grayscale conversion, Gaussian blurring, and Adaptive Thresholding before extracting text using Google's Tesseract OCR engine.
2. **Object Detection Pipeline:** Converts input images into 4D Blobs (`cv2.dnn.blobFromImage`) and passes them through a MobileNet-SSD Caffe model to obtain bounding box coordinates $(X, Y, W, H)$ and object classifications.

---

## 🛠️ Requirements & Installation

### 1. Prerequisites

* **Python 3.8+**
* **Tesseract OCR Engine** installed on your system.
  * **Windows:** Download the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
  * **Linux:** `sudo apt-get install tesseract-ocr`
  * **macOS:** `brew install tesseract`

### 2. Python Dependencies

Install the required Python packages using `pip`:

```bash
pip install opencv-python numpy pytesseract
