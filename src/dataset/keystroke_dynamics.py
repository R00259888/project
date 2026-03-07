import os

import numpy as np
import pandas as pd
import tqdm

class KeystrokeSequence:
    def __init__(self, subject_id, vector):
        self.subject_id = subject_id
        self.__vector = vector

    def vectorise(self):
        return self.__vector

def __load_keystroke_dynamics_dataset(file_path, end_index, time_series):
    df = pd.read_csv(file_path)
    feature_columns = df.columns[3:end_index]

    dataset = []
    for _, instance in tqdm.tqdm(df.iterrows(), total=len(df)):
        subject_id = int(instance.iloc[0][1:])
        vector = instance[feature_columns].to_numpy(dtype=np.float32)
        if time_series: vector = vector.reshape(-1, 1)

        dataset.append(KeystrokeSequence(subject_id, vector))

    return dataset

def load_keyrecs_dataset():
    file_path = os.path.join("datasets", "KeyRecs", "fixed-text.csv")
    return __load_keystroke_dynamics_dataset(file_path, -1, True)

def load_keystroke_dynamics_benchmark_dataset():
    file_path = os.path.join("datasets", "KeystrokeDynamicsBenchmarkDataset", "DSL-StrongPasswordData.csv")
    return __load_keystroke_dynamics_dataset(file_path, None, False)
