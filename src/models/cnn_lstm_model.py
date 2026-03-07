import tensorflow.keras.layers
import tensorflow.keras.preprocessing.sequence

from .base_biometric_model import BaseBiometricModel
from .lstm_model import LSTMModel

class Conv1DWithMasking(tensorflow.keras.layers.Conv1D): supports_masking = True

class CNNLSTMModel(BaseBiometricModel):
    def __init__(self, dataset, subject_id):
        layers = CNNLSTMModel.get_layers()
        layers += LSTMModel.get_layers()
        super().__init__(layers, dataset, subject_id)

    @staticmethod
    def get_layers():
        return [
            tensorflow.keras.layers.Masking(mask_value=0.0),

            Conv1DWithMasking(filters=16, kernel_size=3, padding="same"),
            tensorflow.keras.layers.AveragePooling1D(pool_size=2, padding="same"),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.Activation("relu"),
            tensorflow.keras.layers.Dropout(0.1),

            Conv1DWithMasking(filters=32, kernel_size=3, padding="same"),
            tensorflow.keras.layers.AveragePooling1D(pool_size=2, padding="same"),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.Activation("relu"),
            tensorflow.keras.layers.Dropout(0.1)
        ]

    def prepare_features(self, dataset):
        X = [sequence.vectorise() for sequence in dataset]
        X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, dtype="float32", padding="post")
        y = [
            1 if sequence.subject_id == self.subject_id else 0
            for sequence in dataset
        ]
        return X, y
