"""
Feature extraction using AutoEncoder (using Keras and Tensorflow).
"""
import logging

import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Dense, Reshape

from sklearn.model_selection import train_test_split

from src.pre_stage import PreProcessing, Vectorization


class Autoencoder(Model):

    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
                        layers.Flatten(),
                        layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
                            layers.Dense(784, activation='sigmoid'),
                            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded


if __name__ == "__main__":

    logging.getLogger()

    pre = PreProcessing()
    vec = Vectorization()

    # train and test sets

    data = pre.txt_preprocess(
        file_link='C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt')
    random.shuffle(data)

    X, words = vec.vec_TF_IDF(cleaned_words=data)
    # X = np.array(X.transpose())
    print(type(X.toarray()), X.toarray())

    X_train, X_test, y_train, y_test = train_test_split(X, words)

    latent_dim = 3

    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(X_train, X_train,
                    epochs=10,
                    shuffle=True
                    )

