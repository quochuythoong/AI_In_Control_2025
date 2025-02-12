import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ======================
# STEP 1: TRAIN THE SVM
# ======================

# Load your synthetic fruit dataset (make sure fruit.csv is in your working directory)
data = pd.read_csv('fruits.csv')

# Create feature matrix X and labels y
X = data[['width', 'height', 'color']]
y = data['label']

# Scale features (this is important so that real-time features are scaled in the same way)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM classifier (using one-vs-rest approach)
svm_classifier = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
svm_classifier.fit(X_scaled, y)

# Define a mapping from fruit label to a drawing color (BGR format for OpenCV)
fruit_colors = {
    'apple': (0, 0, 255),    # Red
    'orange': (0, 165, 255), # Orange
    'banana': (0, 255, 255)  # Yellow
}

# ======================
# STEP 2: SET UP WEBCAM STREAMING
# ======================

# Open a connection to your webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space to help extract color information
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ------------------------------
    # STEP 3: DETECT FRUIT-LIKE REGIONS
    # ------------------------------
    #
    # For this demonstration, we create a generic mask that should capture regions
    # with enough color saturation and brightness. This is a very basic method.
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Optionally, clean up the mask with some morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Filter out small contours that are likely noise
        if cv2.contourArea(cnt) < 500:
            continue

        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Extract the ROI (Region Of Interest) from the HSV image
        roi = hsv[y:y+h, x:x+w]

        # Compute the average hue in the ROI.
        # (In OpenCV, hue values range from 0 to 179. We normalize it to 0-1 for our classifier.)
        if roi.size == 0:
            continue
        avg_hue = np.mean(roi[:, :, 0])
        avg_hue_normalized = avg_hue / 179.0

        # ------------------------------
        # STEP 4: EXTRACT FEATURES & PREDICT
        # ------------------------------
        #
        # Here we create a feature vector:
        # - w: bounding box width (in pixels)
        # - h: bounding box height (in pixels)
        # - avg_hue_normalized: average hue (normalized to [0, 1])

        # Create a DataFrame with the same column names as used in training
        features = pd.DataFrame([[w, h, avg_hue_normalized]], columns=['width', 'height', 'color'])
        features_scaled = scaler.transform(features)

        # Use the trained SVM to predict the fruit type
        prediction = svm_classifier.predict(features_scaled)
        fruit_label = prediction[0]

        # Determine the drawing color for this fruit based on our mapping.
        box_color = fruit_colors.get(fruit_label, (255, 255, 255))  # default to white if not found

        # ------------------------------
        # STEP 5: DRAW THE BOUNDING BOX AND LABEL
        # ------------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(frame, fruit_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    # Display the result
    cv2.imshow('Fruit Detection', frame)

    # Press 'q' to exit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
