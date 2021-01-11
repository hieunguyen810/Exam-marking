import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os.path
from os import path
from emnist import extract_test_samples
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import image_processing
import detection
from tensorflow.keras import datasets, layers, models
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
def image_to_string():
    letters = ['A', 'B', 'C', 'D']
    images, labels = extract_test_samples('letters')
    #images = images[:3200]
    #labels = labels[:3200]
    A = images[:400]
    A_label = labels[:400]
    B= images[800:1200]
    B_label = labels[800:1200]
    C = images[1600:2000]
    C_label = labels[1600:2000]
    D = images[2400:2800]
    D_label = labels[2400:2800]
    images = np.vstack([A, B, C, D])
    labels = np.hstack((A_label, B_label, C_label, D_label))
    labels = labels - 1
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train, X_test, X_val = X_train[..., np.newaxis]/255.0, X_test[..., np.newaxis]/255.0, X_val[..., np.newaxis]/255.0
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1, )))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(4, activation= 'softmax'))
    model.summary()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.005),
        metrics=['accuracy'],
    )
    if path.exists("./model.h5"):
            h = model.load_weights("./model.h5")
    else:
        h = model.fit(X_train, y_train, epochs=10, validation_data =(X_val, y_val))
        model.save("./model.h5")
    model.evaluate(X_test, y_test, verbose = 1)
    #y_pred = model.predict(X_test)
    detection.detect_letter()
    n = 1
    predicted_array = np.array([])
    while n<51:
        filename = "./Letter/%s.png" % n
        image = Image.open(filename)
        image = image.resize((28, 28))
        #image = ImageOps.grayscale(image)
        image = np.array(image)
        image = 255 - image
        #image = image_processing.thresholding(image)
        image = cv2.dilate(image, (5, 5))
        image = cv2.dilate(image, (5, 5))
        image = np.reshape(image, [1, 28, 28, 1])
        y_pred = model.predict(image)
        number = np.argmax(y_pred) 
        #print(letters[number])
        n+=1
        predicted_array = np.append(predicted_array, number)
    # epochs_range = range(10)
    # _, (ax1, ax2) = plt.subplots(2)
    # ax1.plot(epochs_range, h.history['loss'], label='Test Loss')
    # ax1.plot(epochs_range, h.history['val_loss'], label='Validation loss')
    # ax1.legend(loc='upper right')
    # ax1.set_title('Loss')
    # ax2.plot(epochs_range, h.history['acc'], label='Test accuracy')
    # ax2.plot(epochs_range, h.history['val_acc'], label='Validation accuracy')
    # ax2.set_title('Accuracy')
    # ax2.legend(loc='upper right')
    # plt.show()
    return predicted_array
# def image_to_number():
#     return student_id
#image = np.reshape(image, [28, 28])
#plt.imshow(image, cmap='gray')
#plt.show()
# predicted_array = image_to_string()
# print(predicted_array)
 