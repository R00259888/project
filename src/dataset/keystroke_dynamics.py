import os

import numpy as np
import sklearn.decomposition

class KeystrokeSequence:
    def __init__(self, subject_id, file_path):
        self.subject_id = subject_id
        self.file_path = file_path
        self.vector = self.__vectorise(self.load_from_file(file_path))

    def load_from_file(self, file_path):
        keystroke_sequence = {}

        with open(file_path) as file_obj:
            scancode_lines = file_obj.read().strip().split("\n")
            for scancode_line in scancode_lines[1:]:
                first_comma_index = scancode_line.find(",")
                if first_comma_index != -1:
                    scancodes = scancode_line[:first_comma_index]
                    times = np.fromstring(scancode_line[first_comma_index + 1:], dtype=np.float32, sep=",")
                else:
                    scancodes = scancode_line
                    times = np.empty(0, dtype=np.float32)

                keystroke_sequence[scancodes] = times

        return keystroke_sequence

    def __truncated_svd(self, np_array, n_components):
        np_array = np_array.reshape(n_components, -1)
        return sklearn.decomposition.TruncatedSVD(n_components=n_components).fit_transform(np_array)

    def __vectorise(self, keystroke_sequence):
        times_items = list(keystroke_sequence.items())
        means = np.zeros(len(times_items), dtype=np.float32)
        medians = np.zeros(len(times_items), dtype=np.float32)

        for i, (_, times) in enumerate(times_items):
            if len(times) > 0:
                means[i] = times.mean()
                medians[i] = np.median(times)

        n_components = 12
        means = self.__truncated_svd(means, n_components)
        medians = self.__truncated_svd(medians, n_components)
        return np.concatenate([means.ravel(), medians.ravel()])

    def vectorise(self):
        return self.vector

def load_keystroke_dynamics_dataset():
    dataset = []
    dataset_path = os.path.join("datasets", "IKDD", "IKDD")

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(dataset_path, file_name)
            subject_id = int(file_name.split("_user")[1].split("_")[0])
            dataset.append(KeystrokeSequence(subject_id, file_path))

    return dataset
