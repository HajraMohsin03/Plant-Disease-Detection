import cv2
import numpy as np
from skimage import feature, io, transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

def extract_color_histogram(image):
    # Convert the image to 8-bit unsigned integer depth
    image = cv2.convertScaleAbs(image)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate color histogram features
    hist_features = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

    return hist_features

# Function to extract LBP features
def extract_lbp_features(gray_region):
    # Calculate LBP features
    lbp_features = feature.local_binary_pattern(gray_region, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 10 + 1), range=(0, 10))
    return hist

# Function to extract features from a segmented region
def extract_features(segmented_region):
    # Convert the segmented region to grayscale
    gray_region = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2GRAY)

    # Calculate statistical features
    mean_intensity = np.mean(gray_region)
    std_dev_intensity = np.std(gray_region)

    # Extract HOG features
    hog_features = feature.hog(gray_region, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

    # Extract LBP features
    lbp_features = extract_lbp_features(gray_region)

    # Extract Color Histogram features
    color_hist_features = extract_color_histogram(segmented_region)

    # Convert mean and std_dev to one-dimensional arrays
    mean_intensity = np.array([mean_intensity])
    std_dev_intensity = np.array([std_dev_intensity])

    # Combine features into a vector
    feature_vector = np.concatenate([mean_intensity, std_dev_intensity, hog_features, color_hist_features, lbp_features])

    return feature_vector

# Function to extract features from a segmented region
def extract_features_from_contour(contour_image):
    # Convert the segmented region to grayscale
    gray_region = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    # Calculate statistical features
    mean_intensity = np.mean(gray_region)
    std_dev_intensity = np.std(gray_region)

    # Extract HOG features
    hog_features = feature.hog(gray_region, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

    # Extract LBP features
    lbp_features = extract_lbp_features(gray_region)

    # Extract Color Histogram features
    color_hist_features = extract_color_histogram(contour_image)

    # Convert mean and std_dev to one-dimensional arrays
    mean_intensity = np.array([mean_intensity])
    std_dev_intensity = np.array([std_dev_intensity])

    # Combine features into a vector
    feature_vector = np.concatenate([mean_intensity, std_dev_intensity, hog_features, color_hist_features, lbp_features])

    return feature_vector

# Function to perform image segmentation
def perform_segmentation(image):
    # Ensure that the image has the correct color depth (np.uint8)
    image = cv2.convertScaleAbs(image)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV thresholds for the leaf and spots
    lower_bound = np.array([20, 50, 50])
    upper_bound = np.array([80, 255, 255])

    # Create a mask for the leaf and spots
    leaf_spot_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply morphological operations to refine the leaf and spot segmentation
    kernel = np.ones((5, 5), np.uint8)
    leaf_spot_mask = cv2.morphologyEx(leaf_spot_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the leaf and spot mask
    contours, _ = cv2.findContours(leaf_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) based on minimum contour area
    min_contour_area = 1000  # Adjust as needed
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    # Draw contours on the original image to highlight the segmented leaf and spots
    contour_image = image.copy()
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)

    # Return the segmented image
    return contour_image


# Function to extract histogram-based features
def extract_histogram_features(image):
    # Convert the image to 8-bit unsigned integer depth
    image = cv2.convertScaleAbs(image)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate color histogram features
    hist_features = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

    return hist_features

# Load the Random Forest model from Google Drive
model_file_path = '/content/drive/MyDrive/random_forest_model.pkl'
loaded_model = load(model_file_path)

# Function to predict using the loaded model
def predict_rf(image_features):
    # Perform predictions using the loaded Random Forest model
    prediction = loaded_model.predict(image_features)
    return prediction

def create_feature_space(image):
  image = transform.resize(image, (64, 64), mode='constant')
  hist_features = extract_histogram_features(image)
  contour_image = perform_segmentation(image)
  feature_vector = extract_features_from_contour(contour_image)
  X_combined = np.concatenate([hist_features, feature_vector], axis=0)
  return X_combined
  