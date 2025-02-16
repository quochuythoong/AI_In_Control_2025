import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ========================================
# STEP 1: TRAIN THE CLASSIFIERS (SVM & KNN)
# ========================================

# Load your synthetic fruit dataset (make sure fruits.csv is in your working directory)
data = pd.read_csv('fruits.csv')

# Create feature matrix X and labels y
X = data[['width', 'height', 'color']]
y = data['label']

# Scale features (this is important so that real-time features are scaled in the same way)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM classifier (one-vs-rest approach)
svm_classifier = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
svm_classifier.fit(X_scaled, y)

# Also, train a KNN classifier (using k=3 here; adjust as needed)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_scaled, y)

# Define a mapping from fruit label to a drawing color (BGR format for OpenCV)
fruit_colors = {
    'apple': (0, 0, 255),    # Red
    'orange': (0, 165, 255), # Orange
    'banana': (0, 255, 255)  # Yellow
}

# ========================================
# GLOBAL VARIABLES
# ========================================
# detections: list of candidate fruit regions detected in the current frame.
# Each detection is a tuple: (x, y, w, h, features)
detections = []
# clicked_points: list of clicked points with KNN label.
# Each element is a tuple: (click_x, click_y, knn_label)
clicked_points = []

# ========================================
# STEP 2: MOUSE CALLBACK FOR KNN CLASSIFICATION
# ========================================
def on_mouse(event, x, y, flags, param):
    global detections, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if not detections:
            print("No candidate regions available to classify!")
            return
        
        # Find the candidate detection whose center is closest to the clicked point.
        distances = []
        for (bx, by, bw, bh, features) in detections:
            center_x = bx + bw / 2
            center_y = by + bh / 2
            dist = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
            distances.append((dist, features))
        
        # Get the candidate with the smallest distance.
        distances.sort(key=lambda item: item[0])
        nearest_features = distances[0][1]
        
        # Prepare the feature vector (as a DataFrame to keep column names)
        feat_df = pd.DataFrame([nearest_features], columns=['width', 'height', 'color'])
        feat_scaled = scaler.transform(feat_df)
        
        # Use the KNN classifier to predict the fruit label for this candidate.
        knn_label = knn_classifier.predict(feat_scaled)[0]
        print(f"Clicked at ({x}, {y}). KNN prediction: {knn_label}")
        
        # Save the clicked point and the KNN label so we can display it.
        clicked_points.append((x, y, knn_label))

# ========================================
# STEP 3: SET UP WEBCAM STREAMING AND DETECTION
# ========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

window_name = 'Fruit Detection with SVM & KNN'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Clear the candidate detections for this frame.
    detections = []

    # Convert frame to HSV color space for better color segmentation.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ------------------------------
    # STEP 3a: DETECT FRUIT-LIKE REGIONS
    # ------------------------------
    # Create a mask that captures regions with enough saturation and brightness.
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up the mask with morphological operations.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the contours.
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # filter out small regions
            continue
        
        # Get bounding box for the contour.
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Extract the ROI from the HSV image.
        roi = hsv[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        
        # Compute the average hue in the ROI (normalize to [0,1])
        avg_hue = np.mean(roi[:, :, 0])
        avg_hue_normalized = avg_hue / 179.0
        
        # Create a feature vector: [width, height, avg_hue_normalized]
        features = [w, h, avg_hue_normalized]
        
        # Save the detection for use in the mouse callback.
        detections.append((x, y, w, h, features))
        
        # ------------------------------
        # STEP 4: PREDICT USING THE SVM FOR DISPLAY
        # ------------------------------
        # Convert feature vector to a DataFrame to maintain column names.
        feat_df = pd.DataFrame([features], columns=['width', 'height', 'color'])
        feat_scaled = scaler.transform(feat_df)
        svm_prediction = svm_classifier.predict(feat_scaled)[0]
        
        # Get the corresponding drawing color.
        box_color = fruit_colors.get(svm_prediction, (255, 255, 255))
        
        # Draw the bounding box and label as predicted by SVM.
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(frame, svm_prediction, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
    
    # ------------------------------
    # STEP 5: DRAW THE KNN CLICKED POINTS
    # ------------------------------
    for (click_x, click_y, knn_label) in clicked_points:
        # Mark the clicked point with a small green circle.
        cv2.circle(frame, (click_x, click_y), 5, (0, 255, 0), -1)
        # Write the KNN predicted label near the clicked point.
        cv2.putText(frame, knn_label, (click_x + 10, click_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display the frame.
    cv2.imshow(window_name, frame)
    
    # Press 'q' to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: release webcam and close windows.
cap.release()
cv2.destroyAllWindows()
