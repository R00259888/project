import os

import numpy as np
import tqdm

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

    def __vectorise(self, keystroke_sequence):
        feature_count = 256
        feature_vector = np.zeros(feature_count * 2, dtype=np.float32)
        for i in range(feature_count):
            if (key := str(i) + "-0") in keystroke_sequence:
                if len(keystroke_sequence[key]) != 0: # Avoid div by zero exception.
                    feature_vector[i * 2] = keystroke_sequence[key].mean()
                    feature_vector[(i * 2) + 1] = np.std(keystroke_sequence[key])
        return feature_vector

    def vectorise(self):
        return self.vector

def load_ikdd_keystroke_dynamics_dataset():
    dataset = []
    dataset_path = os.path.join("datasets", "IKDD", "IKDD")

    for file_name in tqdm.tqdm(os.listdir(dataset_path)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(dataset_path, file_name)
            subject_id = int(file_name.split("_user")[1].split("_")[0])
            dataset.append(KeystrokeSequence(subject_id, file_path))

    return dataset
