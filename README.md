/// This README.md file explains the overall flow and mechanism of the project in simple, non-technical terms, covering everything from data preparation and SVM training to live video processing and fruit detection. ///

Press 'Ctrl + Shift + v' to preview this README.md 

---

# Fruit Detection with SVM and Webcam Streaming

## Overview

This project combines machine learning with real-time computer vision to detect and classify three types of fruit—apple, orange, and banana—using a webcam. The system uses a Support Vector Machine (SVM) classifier trained on a synthetic dataset of fruit measurements. Once trained, the program processes the live video feed from the webcam, identifies candidate regions that might contain fruit, extracts basic features from those regions, and then predicts which type of fruit is present. The result is shown on the screen with a colored bounding box and a label that indicates the fruit type.

## Program Manual (How to use)

- Run the main.py
- Show the fruit: Apple, Banana, Orange
- The program detects the fruit (lighting is crucial, make sure you have a good light setting)
- Press 'q' to exit the program 

## Project Flow

1. **Dataset Creation and Preparation:**
   - A synthetic dataset is created, representing fruit samples with three features: width, height, and a normalized color value (derived from hue).
   - Each fruit (apple, orange, and banana) is given its own set of measurements that simulate real-world differences (e.g., bananas are longer and have different color characteristics compared to apples and oranges).
   - This dataset is stored in a CSV file, which serves as the foundation for training the classifier.

2. **Training the SVM Classifier:**
   - The SVM is a supervised machine learning algorithm used for classification.
   - The program reads the dataset, extracts the features (width, height, and color) and corresponding fruit labels, and then standardizes these features so that they are on a similar scale.
   - With this prepared data, the SVM is trained to learn how to differentiate between the three types of fruit. Essentially, the algorithm finds the best boundaries (or decision surfaces) that separate apples, oranges, and bananas based on their measured characteristics.

3. **Real-Time Webcam Streaming and Detection:**
   - The program activates the computer’s webcam to start capturing live video.
   - Each video frame is processed to convert it into a color space that is more useful for detecting fruit (for example, transforming the image to highlight hue and saturation).
   - A simple segmentation technique is applied to each frame to isolate regions that could contain fruit. This involves filtering out parts of the image that don’t have the expected color intensity or saturation.
   - Contours (or outlines) are then detected in these candidate regions. Each contour is considered as a potential fruit.

4. **Feature Extraction and Classification in Real-Time:**
   - For every detected region, the program measures the size of the region (using width and height of the bounding box) and calculates the average color information.
   - These measurements are packaged into a feature vector that matches the format used during the training phase.
   - The SVM classifier then uses these features to predict which type of fruit is in the region.
   - The predicted fruit type is mapped to a specific color (e.g., red for apple, orange for orange, yellow for banana).

5. **Visualization and Output:**
   - For each detected fruit, the program draws a colored bounding box around the region in the live video feed.
   - A label indicating the fruit type is displayed near the bounding box.
   - The result is a live video stream where fruit are highlighted and identified in real time.

## Mechanism in Simple Terms

- **Learning the Differences:**  
  The SVM is trained on examples that tell it how apples, oranges, and bananas differ in size and color. Think of it like teaching someone to tell fruits apart by showing many examples.

- **Seeing and Deciding:**  
  When the program is running, it watches the video feed and looks for parts of the image that might be fruit. It then measures these parts, compares them with what it has learned, and decides which fruit they most closely resemble.

- **Showing the Results:**  
  Once a decision is made, the program highlights the fruit in the video by drawing a box around it and writing its name in a matching color. This way, you can see which fruit is detected in real time.

## Conclusion

This project demonstrates a basic but complete pipeline—from creating a dataset and training a classifier, to processing live video input and visualizing the results. It provides a practical example of how machine learning can be combined with computer vision techniques to perform real-time object detection and classification.
