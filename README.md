# Number Plate Detection System

A deep learning-based application that detects license plates in images and video streams, performs OCR to extract the plate numbers, and logs the results in real-time.

![License Plate Detection Demo](https://raw.githubusercontent.com/faseehahmed26/Number-Plate-Detection/main/sample_output.jpg)

## Overview

This project uses the TensorFlow Object Detection API with a custom-trained SSD MobileNet v2 model to detect license plates in images and video. Once detected, the system uses EasyOCR to recognize and extract the alphanumeric characters from the license plates.

## Features

- **License Plate Detection**: Utilizes transfer learning with SSD MobileNet v2 FPN-Lite architecture to detect license plates with high accuracy
- **Optical Character Recognition**: Implements EasyOCR to extract text from detected license plates
- **Real-time Processing**: Processes webcam feed in real-time, detecting and recognizing license plates on the fly
- **Result Logging**: Automatically saves detection results to CSV files with unique identifiers
- **Multi-platform Support**: Exports models to TensorFlow, TFJS, and TFLite formats for deployment on various platforms

## Project Structure

- **1. Image Collection.ipynb**: Notebook for collecting and labeling training images
- **2. Training and Detection.ipynb**: Main notebook for training the model and performing detection
- **Detection_images/**: Directory containing cropped license plate images
- **Tensorflow/**: Directory containing model files, training data, and configuration
- **detection_results.csv**: Log file for static image detection results
- **realtime_results.csv**: Log file for real-time webcam detection results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/faseehahmed26/Number-Plate-Detection.git
   cd Number-Plate-Detection
   # Install the required packages:
   pip install tensorflow tensorflow-gpu opencv-python easyocr matplotlib numpy uuid
    ```

Model Architecture
The project uses a custom-trained SSD MobileNet v2 FPN-Lite model with the following specifications:

Base model: SSD MobileNet v2 FPN-Lite 320x320
Transfer learning: Fine-tuned from COCO pre-trained weights
Input size: 320x320 pixels
Output: Detection boxes, classes, and confidence scores

Performance

Detection accuracy: >90% on test dataset
OCR accuracy: ~85% (varies based on image quality and lighting conditions)
Processing speed: 5-10 FPS on CPU, 15-30 FPS on GPU (depends on hardware)

License
This project is licensed under the MIT License - see the LICENSE file for details.
