import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2hsv
from skimage import io
import os

def compute_sift_features(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift_detector = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    keypoints, descriptors = sift_detector.detectAndCompute(gray_image, None)
    
    return keypoints, descriptors

def compute_hsv_features(image):
    # Convert image to HSV
    hsv_image = rgb2hsv(image)
    
    # Compute average hue and saturation
    avg_hue = np.mean(hsv_image[:, :, 0])
    avg_saturation = np.mean(hsv_image[:, :, 1])
    
    return avg_hue, avg_saturation

def visualize_dataset(image_folder):
    # Print directory contents for debugging
    print("Files in directory:")
    try:
        for filename in os.listdir(image_folder):
            print(filename)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # List all .jpg files in the directory
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    num_images = len(image_files)
    
    if num_images == 0:
        print("No images found in the directory.")
        return
    
    # Handle the case of a single image
    if num_images == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        
        # Compute SIFT features
        keypoints, descriptors = compute_sift_features(image)
        
        # Compute HSV features
        avg_hue, avg_saturation = compute_hsv_features(image)
        
        # Display image
        axes[i].imshow(image)
        axes[i].set_title(f'Hue: {avg_hue:.2f}\nSaturation: {avg_saturation:.2f}')
        axes[i].axis('off')
    
    plt.show()

# Path to your image dataset
image_folder = './jpg'
visualize_dataset(image_folder)

