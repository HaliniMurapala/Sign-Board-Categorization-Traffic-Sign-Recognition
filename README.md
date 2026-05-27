# Sign Board Categorization — Traffic Sign Recognition

A deep learning project I built as my final year B.Tech 
project at Aditya Institute of Technology and Management. 
The system uses a Convolutional Neural Network to classify 
43 different types of traffic signs from images and 
real time webcam feed.

---

## Why I Built This

Traffic sign recognition is an important part of modern 
driver assistance systems and self driving cars. I wanted 
to understand how deep learning can be applied to real 
world computer vision problems. This project helped me 
learn CNN architecture, image preprocessing and 
model deployment using TensorFlow.

---

## What It Does

- Classifies 43 different types of traffic signs
- Works on both static images and live webcam feed
- Shows sign name with confidence percentage
- Displays top 3 predictions for each input
- Trained on GTSRB dataset with 39000+ images
- Real time detection using OpenCV webcam

---

## How It Works

The system has 3 main parts:

1. Training (Train.py)
   - Loads GTSRB dataset from local folder
   - Resizes all images to 32x32 pixels
   - Normalizes pixel values to 0-1
   - Trains CNN model with 3 convolutional blocks
   - Uses early stopping to avoid overfitting
   - Saves best model automatically

2. Image Testing (Test_image.py)
   - Loads any traffic sign image
   - Preprocesses and feeds to trained model
   - Shows predicted sign name and confidence
   - Displays top 3 predictions in terminal

3. Webcam Testing (Test_webcam.py)
   - Opens live webcam feed
   - Processes frames in real time
   - Shows detected sign name on screen
   - Only displays result above 70% confidence

---

## CNN Model Architecture

Block 1:
- Conv2D 32 filters + Batch Normalization
- MaxPooling + Dropout 25%

Block 2:
- Conv2D 64 filters + Batch Normalization
- MaxPooling + Dropout 25%

Block 3:
- Conv2D 128 filters + Batch Normalization
- MaxPooling + Dropout 25%

Fully Connected:
- Dense 512 neurons + Batch Normalization
- Dropout 50%
- Output 43 classes with Softmax

---

## Dataset

- Name: GTSRB German Traffic Sign Recognition Benchmark
- Total images: 39000+
- Number of classes: 43 traffic sign types
- Image size: 32x32 pixels
- Source: Kaggle GTSRB dataset

---

## Technologies Used

- Python 3
- TensorFlow and Keras
- OpenCV
- NumPy
- Scikit-learn
- Pandas

---

## How to Run

Step 1 — Install dependencies:
pip install tensorflow opencv-python numpy pandas scikit-learn

Step 2 — Download GTSRB dataset from Kaggle
and place Train folder in project directory

Step 3 — Train the model:
python Train.py

Step 4 — Test with an image:
python Test_image.py imagefortest.png

Step 5 — Test with webcam:
python Test_webcam.py

---

## Sample Output

Terminal output:
Sign       : General Caution
Confidence : 93.60%
Class ID   : 18
Top 3 Predictions:

General Caution    — 93.60%
Road Work          — 4.20%
Pedestrians        — 1.10%

## Demo Screenshots

![Output 1](Demo_Output/output1.png)
![Output 2](Demo_Output/output2.png)
![Output 3](Demo_Output/output1.png)
![Output 5](Demo_Output/output2.png)

## Project Structure
Sign-Board-Categorization/
├── archive/
│   └── Train/
├── model/
│   └── traffic_sign_model.h5
├── Demo_Output/
│   ├── output1.png
│   └── output2.png
├── Train.py
├── Test_image.py
├── Test_webcam.py
├── labels.py
└── README.md

---

## What I Learned

- How CNN layers extract features from images
- Importance of data normalization and augmentation
- How batch normalization stabilizes training
- How dropout prevents overfitting
- Real time image processing using OpenCV
- Saving and loading trained models

---

## Future Improvements

- Add data augmentation for better accuracy
- Deploy as web application using Flask
- Add Indian traffic sign dataset
- Optimize for mobile using TensorFlow Lite

---

## About Me

Murapala Halini
Embedded Systems Engineer — Fresher
Bengaluru, India
LinkedIn: linkedin.com/in/halinimurapala
GitHub: github.com/HaliniMurapala
