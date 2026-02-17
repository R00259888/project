import tensorflow.keras.layers
import tensorflow.keras.preprocessing.sequence

from .base_biometric_model import BaseBiometricModel

class MouseDynamicsLSTMModel(BaseBiometricModel):
    def __init__(self, dataset, subject_id):
        layers = [tensorflow.keras.layers.Masking(mask_value=0.0)]
        layers += MouseDynamicsLSTMModel.get_layers()
        super().__init__(layers, dataset, subject_id)

    @staticmethod
    def get_layers():
        return [
            tensorflow.keras.layers.LSTM(64, recurrent_activation="sigmoid", use_cudnn=False),
            tensorflow.keras.layers.Activation("tanh"),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.Dropout(0.1),
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dropout(0.1),
            tensorflow.keras.layers.Dense(1, activation="sigmoid", dtype="float32")
        ]

    def prepare_features(self, dataset):
        X = [mouse_event_sequence.vectorise() for mouse_event_sequence in dataset]
        X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, dtype="float32", padding="post")
        y = [
            1 if mouse_event_sequence.subject_id == self.subject_id else 0
            for mouse_event_sequence in dataset
        ]
        return X, y
