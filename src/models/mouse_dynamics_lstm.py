import tensorflow.keras.layers
import tensorflow.keras.models

class MouseDynamicsLSTMModel:
    def __init__(self, dataset):
        self.sequential = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.LSTM(1)
        ])
        self.sequential.compile(optimizer="adam", loss="mse")
        self.dataset = dataset

    def fit(self, epochs):
        pass
