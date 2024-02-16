import cv2
import tensorflow as tf
import numpy as np

# Load pre-trained CNN model
model = tf.keras.models.load_model('your_model_path')

# Define categories
categories = ['water bottle', 'pencil case', 'coat', 'watch', 'key', 'uniform', 'umbrella', 'glasses', 'hat', 'gloves', 'shoes', 'earphones', 'others']

# Function to preprocess input image
def preprocess_image(image):
    # Preprocess image (resize, normalize, etc.)
    resized_image = cv2.resize(image, (224, 224))  # Adjust size as per model requirements
    normalized_image = resized_image / 255.0  # Normalize pixel values
    return normalized_image.reshape(1, 224, 224, 3)  # Reshape for model input

# Function to predict category
def predict_category(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    category_index = np.argmax(prediction)
    return categories[category_index]

# Initialize camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    ret, frame = camera.read()
    if not ret:
        break

    # Display captured frame
    cv2.imshow('Lost and Found Camera', frame)

    # Perform object categorization
    category = predict_category(frame)
    print('Predicted category:', category)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
