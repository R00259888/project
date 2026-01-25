import numpy as np
import tensorflow.keras.layers
import tensorflow.keras.models

class KeystrokeDynamicsNNModel:
    def __init__(self, dataset):
        self.sequential = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Dense(128, activation="relu"),
            tensorflow.keras.layers.Dense(64, activation="relu"),
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dense(1)
        ])
        self.sequential.compile(optimizer="adam", loss="mse")
        self.dataset = dataset

    def fit(self, epochs):
        X = np.array([keystroke_sequence.vectorise() for keystroke_sequence in self.dataset])
        y = np.array([keystroke_sequence.subject_id for keystroke_sequence in self.dataset])
        self.sequential.fit(X, y, epochs=epochs)
