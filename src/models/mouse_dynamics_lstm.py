import numpy as np
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.preprocessing.sequence

class MouseDynamicsLSTMModel:
    def __init__(self, dataset, subject_id):
        self.sequential = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.LSTM(64),
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dense(1)
        ])
        self.sequential.compile(optimizer="adam", loss="mse")
        self.dataset = dataset
        self.subject_id = subject_id

    def prepare_features(self, dataset):
        X = [mouse_event_sequence.vectorise() for mouse_event_sequence in dataset]
        X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, dtype="float32", padding="post")
        y = [
            1 if mouse_event_sequence.subject_id == self.subject_id else 0
            for mouse_event_sequence in dataset
        ]
        return X, y

    def fit(self, epochs):
        X, y = self.prepare_features(self.dataset)
        self.sequential.fit(X, np.array(y), epochs=epochs)

    def predict(self, X):
        return self.sequential.predict(X)
