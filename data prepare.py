import os
import cv2
import numpy as np

# Define categories and corresponding labels
categories = ['water_bottle', 'pencil_case', 'coat', 'watch', 'key', 'uniform', 'umbrella', 'glasses', 'hat', 'gloves', 'shoes', 'earphones', 'others']
num_classes = len(categories)

# Initialize lists to store image data and labels
data = []
labels = []

# Loop through each category folder
for label, category in enumerate(categories):
    folder_path = f'path/to/{category}/'  # Replace with actual path to the category folder
    for img_filename in os.listdir(folder_path):
        if img_filename.endswith('.img'):
            img_path = os.path.join(folder_path, img_filename)
            # Read and preprocess image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Resize image to a consistent size
            img = img / 255.0  # Normalize pixel values
            # Add image data and label to lists
            data.append(img)
            labels.append(label)

# Convert lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split data into train, validation, and test sets (e.g., using train_test_split from scikit-learn)
# Further processing and model training will depend on your specific setup and requirements
