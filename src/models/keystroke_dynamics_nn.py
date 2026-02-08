import numpy as np
import tensorflow.keras.layers

from .base_biometric_model import BaseBiometricModel

class KeystrokeDynamicsNNModel(BaseBiometricModel):
    def __init__(self, dataset, subject_id):
        super().__init__(KeystrokeDynamicsNNModel.get_layers(), dataset, subject_id)

    @staticmethod
    def get_layers():
        return [
            tensorflow.keras.layers.Dense(128, activation="relu"),
            tensorflow.keras.layers.Dense(64, activation="relu"),
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dense(1, activation="sigmoid", dtype="float32")
        ]

    def prepare_features(self, dataset):
        X = np.array([keystroke_sequence.vectorise() for keystroke_sequence in dataset])
        y = [
            1 if keystroke_sequence.subject_id == self.subject_id else 0
            for keystroke_sequence in dataset
        ]
        return X, y
