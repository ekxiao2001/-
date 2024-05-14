#task-start
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def load_data():

    with open("dataset", "rb") as f:
        data = pickle.load(f, encoding="bytes")

        train_data = np.asarray(data[b'data'][:10])
        train_labels = data[b'labels'][:10]

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = to_categorical(train_labels)
    return train_data, train_labels


def build_model_and_train():
    train_images, train_labels = load_data()
    model = Sequential()

    # TODO
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    while True:
        history = model.fit(train_images, train_labels, epochs=5, batch_size=32)
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        if loss < 1e-3 and accuracy == 1.0:
            break
    
    model.save("image_classify.h5")


build_model_and_train()
#task-end