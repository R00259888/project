import numpy as np
import tensorflow.keras.models

from ..defences import adversarial_defence

class BaseBiometricModel(tensorflow.keras.models.Sequential):
    def __init__(self, layers, dataset, subject_id, adversarial_training):
        super().__init__(layers)
        self.compile(optimizer="adam", loss="binary_crossentropy")
        self.dataset = dataset
        self.subject_id = subject_id
        self.adversarial_training = adversarial_training

    def train_step(self, data):
        if not self.adversarial_training:
            return super().train_step(data)
        return adversarial_defence(self, data)

    def prepare_features(self, dataset):
        pass

    def predict(self, X, **kwargs):
        return super().predict(X, batch_size=512, verbose=0, **kwargs)

    def fit(self, epochs, class_weight):
        X, y = self.prepare_features(self.dataset)
        super().fit(X, np.array(y), epochs=epochs, batch_size=512, class_weight=class_weight, verbose=0)