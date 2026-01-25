import os

import numpy as np
import sklearn.decomposition

class KeystrokeSequence:
    def __init__(self, subject_id, file_obj):
        self.subject_id = subject_id
        self.keystroke_sequence = {}

        scancode_lines = file_obj.read().strip().split("\n")
        for scancode_line in scancode_lines[1:]:
            first_comma_index = scancode_line.find(",")
            if first_comma_index != -1:
                scancodes = scancode_line[:first_comma_index]
                times = np.fromstring(scancode_line[first_comma_index + 1:], dtype=np.float32, sep=",")
            else:
                scancodes = scancode_line
                times = np.empty(0, dtype=np.float32)

            self.keystroke_sequence[scancodes] = times

    def __pca(self, np_array, n_components):
        np_array = np_array.reshape(n_components, -1)
        return sklearn.decomposition.TruncatedSVD(n_components=n_components).fit_transform(np_array)

    def vectorise(self):
        times_items = list(self.keystroke_sequence.items())
        means = np.zeros(len(times_items), dtype=np.float32)
        medians = np.zeros(len(times_items), dtype=np.float32)

        for i, (_, times) in enumerate(times_items):
            if len(times) > 0:
                means[i] = times.mean()
                medians[i] = np.median(times)

        n_components = 12
        means = self.__pca(means, n_components)
        medians = self.__pca(medians, n_components)
        return np.concatenate([means.ravel(), medians.ravel()])

def load_keystroke_dynamics_dataset():
    dataset = []
    dataset_path = os.path.join("datasets", "IKDD", "IKDD")

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(dataset_path, file_name)) as file_obj:
                subject_id = int(file_name.split("_user")[1].split("_")[0])
                dataset.append(KeystrokeSequence(subject_id, file_obj))

    return dataset
