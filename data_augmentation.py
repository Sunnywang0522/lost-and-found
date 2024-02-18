import cv2
import os
import numpy as np

# Function to perform data augmentation on images in a folder
def augment_data (folder_path,output_path, num_augmentations_per_image=3):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Loop through each image file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):  # Assuming image files are in .img format
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            # Apply data augmentation
            for i in range(num_augmentations_per_image):
                augmented_img = img.copy()

                # Random rotation (-15 to 15 degrees)
                angle = np.random.randint(-180, 180)
                M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
                augmented_img = cv2.warpAffine(augmented_img, M, (img.shape[1], img.shape[0]))

                # Random stretching (scale factor 0.8 to 1.2)
                scale_factor = np.random.uniform(0.8, 1.2)
                augmented_img = cv2.resize(augmented_img, None, fx=scale_factor, fy=scale_factor)

                # Random horizontal flipping
                if np.random.rand() < 0.5:
                    augmented_img = cv2.flip(augmented_img, 1)  # Flip horizontally

                # Save augmented image
                output_filename = os.path.splitext(filename)[0] + f'_aug{i+1}.jpg'
                output_img_path = os.path.join(output_path, output_filename)
                cv2.imwrite(output_img_path, augmented_img)

# Example usage:
augment_data('/Users/ivycr1/Documents/github/Electronics', '/Users/ivycr1/Documents/github/Electronics', num_augmentations_per_image=3)
