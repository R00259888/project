import numpy as np
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.preprocessing.sequence

class MouseDynamicsLSTMModel:
    def __init__(self, dataset):
        self.sequential = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.LSTM(64),
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dense(1)
        ])
        self.sequential.compile(optimizer="adam", loss="mse")
        self.dataset = dataset

    def fit(self, epochs):
        X = [mouse_event_sequence.vectorise() for mouse_event_sequence in self.dataset]
        X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, dtype="float32", padding="post")
        y = np.array([mouse_event_sequence.subject_id for mouse_event_sequence in self.dataset])
        self.sequential.fit(X, y, epochs=epochs)
