import numpy as np
import tensorflow.keras.layers
import tensorflow.keras.models

class KeystrokeDynamicsNNModel:
    def __init__(self, dataset, subject_id):
        self.sequential = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Dense(128, activation="relu"),
            tensorflow.keras.layers.Dense(64, activation="relu"),
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dense(1, activation="sigmoid")
        ])
        self.sequential.compile(optimizer="adam", loss="binary_crossentropy")
        self.dataset = dataset
        self.subject_id = subject_id

    def prepare_features(self, dataset):
        X = np.array([keystroke_sequence.vectorise() for keystroke_sequence in dataset])
        y = [
            1 if keystroke_sequence.subject_id == self.subject_id else 0
            for keystroke_sequence in dataset
        ]
        return X, y

    def fit(self, epochs):
        X, y = self.prepare_features(self.dataset)
        self.sequential.fit(X, np.array(y), epochs=epochs)

    def predict(self, X):
        return self.sequential.predict(X)
