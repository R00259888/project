import numpy as np
import tensorflow.keras.models

class BaseBiometricModel(tensorflow.keras.models.Sequential):
    def __init__(self, layers, dataset, subject_id):
        super().__init__(layers)
        self.compile(optimizer="adam", loss="binary_crossentropy")
        self.dataset = dataset
        self.subject_id = subject_id

    def prepare_features(self, dataset):
        pass

    def fit(self, epochs):
        X, y = self.prepare_features(self.dataset)
        super().fit(X, np.array(y), epochs=epochs)