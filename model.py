import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

from collections import Counter

PRINT = 0

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

labels_str = ['water_bottle', 'pencil_case', 'coat', 'umbrella', 'watch', 'electronics', 'gloves'] 
def load_data():
    data = np.load("image_np_arr.npy")
    labels = np.load("labels_np_arr.npy")

    if PRINT:
        print(f"Data have been loaded with shape: {data.shape}")
        print(f"Labels have been loaded with shape: {labels.shape}")
        cntr = Counter(labels)
        for ind, key in enumerate(labels_str):
            print(f"{key}: {cntr[ind]}")
    return data, labels

# Define CNN model architecture

def plot_history(history):
    from matplotlib import pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
def run_model(X_train, X_test, y_train, y_test, num_epochs):

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(X_train[0].shape)),
        MaxPooling2D((2, 2)),
        Dropout(.4),
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train[0].shape)),
        MaxPooling2D((2, 2)),
        Dropout(.3),
        Conv2D(16, (3, 3), activation='relu', input_shape=(X_train[0].shape)),
        MaxPooling2D((2, 2)),
        Dropout(.3),
        # layers.Conv2D(64, (3, 3), activation='relu'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=[X_test, y_test])

    plot_history(history)
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)


def main(): 
    data, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(data, labels , test_size=.2)
    if PRINT:
        print(f"Training shape: {X_train.shape}")
    run_model(X_train, X_test, y_train, y_test, 10)

main()