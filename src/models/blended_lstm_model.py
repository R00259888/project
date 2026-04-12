import numpy as np

from .cnn_lstm_model import CNNLSTMModel
from .lstm_model import LSTMModel

class BlendedLSTMModel:
    def __init__(self, dataset, subject_id, adversarial_training):
        self.cnn_lstm = CNNLSTMModel(dataset, subject_id, adversarial_training)
        self.lstm = LSTMModel(dataset, subject_id, adversarial_training)

    def prepare_features(self, dataset):
        return self.lstm.prepare_features(dataset)

    def __call__(self, X, training=False):
        return (self.cnn_lstm(X, training=training) + self.lstm(X, training=training)) / 2

    def compute_loss(self, y, y_pred):
        return self.lstm.compute_loss(y=y, y_pred=y_pred)

    def predict(self, X):
        return np.mean([self.cnn_lstm.predict(X), self.lstm.predict(X)], axis=0)

    def fit(self, epochs, class_weight):
        self.lstm.fit(epochs, class_weight)
        self.cnn_lstm.fit(epochs, class_weight)
