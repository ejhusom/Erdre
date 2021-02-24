#!/usr/bin/env python3
# ============================================================================
# File:     autoencoder.py
# Created:  2020-06-29
# ----------------------------------------------------------------------------
# Description:
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam


class Autoencoder():

    def __init__(self, X_train, X_test):

        self.X_train = X_train
        self.X_test = X_test
        self.input_x = self.X_train.shape[-2]
        self.input_y = self.X_train.shape[-1]

        self.latent_dim = 128
        print(self.X_train.shape)

        self._create_model()


    def _create_model(self):
        """Define the autoencoder model."""

        inputs = layers.Input(shape=(self.input_x, self.input_y))
        x = inputs 
        x = layers.Conv1D(32, 3)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1D(64, 3)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        # x = layers.Conv1D(64, 3)(x)
        # x = layers.LeakyReLU(alpha=0.2)(x)
        volume_size = backend.int_shape(x)
        print(volume_size)
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim)(x)

        self.encoder = models.Model(inputs, latent, name="encoder")

        print(self.encoder.summary())

        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(np.prod(volume_size[1:]))(latent_inputs)
        x = layers.Reshape((volume_size[1], volume_size[2]))(x)

        x = layers.Conv1DTranspose(64, 3)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        # x = layers.Conv1DTranspose(32, 3)(x)
        # x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv1DTranspose(self.input_y, 3)(x)
        outputs = layers.Activation("sigmoid")(x)

        self.decoder = models.Model(latent_inputs, outputs, name="decoder")
        print(self.decoder.summary())

        self.autoencoder = models.Model(
            inputs, self.decoder(self.encoder(inputs)), name="autoencoder"
        )

        self.autoencoder.compile(loss="mse", optimizer=Adam())

    def train(self):
        """Train the autoencoder."""

        epochs = 20

        H = self.autoencoder.fit(
            self.X_train, self.X_train, 
            validation_split=0.8,
            epochs=epochs, batch_size=32,
            verbose=1
        )
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("tmp_ae.png")
        plt.show()


    def test(self):
        """Test autoencoder on test set and plot the result."""

        decoded = self.autoencoder.predict(self.X_test)

        result = np.hstack([self.X_test[0], decoded[0]])

        plt.figure()
        plt.imshow(result)
        plt.savefig("tmp_ae_im.png")
        plt.show()

        mse = ((self.X_test - decoded)**2).mean(axis=0)
        print("MSE autoencoder: {}".format(mse))

    def encode_inputs(self):
        """Encode the train and test set."""

        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)

        return self.X_train, self.X_test

    def denoise_inputs(self):
        """
        Run the train and test set through the autoencoder in order to
        obtain a "denoised" dataset.
        """

        self.X_train = self.autoencoder.predict(self.X_train)
        self.X_test = self.autoencoder.predict(self.X_test)

        return self.X_train, self.X_test

