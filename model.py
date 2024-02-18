import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_data():
    data = np.load("image_np_arr.npy")
    labels = np.load("labels_np_arr.npy")

    print(f"Data have been loaded with shape: {data.shape}")
    print(f"Labels have been loaded with shape: {labels.shape}")
    return data, labels

# Define CNN model architecture

def run_model(X_train, X_test, y_train, y_test):
    return
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=num_epochs,
                        validation_data=(validation_images, validation_labels))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


def main(): 
    data, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(data, labels , test_size=.2 )
    run_model(X_train, X_test, y_train, y_test)

main()