import numpy as np

from .cnn_lstm_model import CNNLSTMModel
from .lstm_model import LSTMModel

class BlendedLSTMModel:
    def __init__(self, dataset, subject_id, adversarial_training):
        self.cnn_lstm = CNNLSTMModel(dataset, subject_id, adversarial_training)
        self.lstm = LSTMModel(dataset, subject_id, adversarial_training)

    def prepare_features(self, dataset):
        return self.lstm.prepare_features(dataset)

    def predict(self, X):
        return np.mean([self.cnn_lstm.predict(X), self.lstm.predict(X)], axis=0)

    def fit(self, epochs, class_weight):
        self.lstm.fit(epochs, class_weight)
        self.cnn_lstm.fit(epochs, class_weight)
