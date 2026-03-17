import dataclasses, functools, os, zipfile

import numpy as np
import pandas as pd
import tqdm

@dataclasses.dataclass
class MouseEvent:
    timestamp: int
    x: int
    y: int
    button_pressed: int
    subject_id: int

class MouseEventSequence:
    CHUNK_SIZE = 128

    def __init__(self, data):
        match data:
            case pd.DataFrame():
                self.data = data[["Timestamp", "X", "Y", "Button Pressed", "Subject ID"]].to_numpy()
                self.subject_id = int(self.data[0, 4])
            case np.ndarray():
                self.data = data
                self.subject_id = int(self.data[0, 4])
            case _:
                self.data = None
                self.mouse_event_sequence = data
                self.subject_id = data[0].subject_id

    def __getattr__(self, name):
        if name == 'mouse_event_sequence':
            self.mouse_event_sequence = [
                MouseEvent(
                    timestamp=int(mouse_event[0]),
                    x=int(mouse_event[1]),
                    y=int(mouse_event[2]),
                    button_pressed=int(mouse_event[3]),
                    subject_id=int(mouse_event[4])
                )
                for mouse_event in self.data
            ]
            return self.mouse_event_sequence
        raise AttributeError(name) # Needed by deepcopy

    def __len__(self):
        if self.data is not None: return len(self.data)
        return len(self.mouse_event_sequence)

    def __getitem__(self, key):
        if isinstance(key, slice):
            if self.data is not None: return MouseEventSequence(self.data[key])
            return MouseEventSequence(self.mouse_event_sequence[key])
        return self.mouse_event_sequence[key]

    def chunkify(self):
        return [self[i:i + self.CHUNK_SIZE] for i in range(0, len(self), self.CHUNK_SIZE)]

    def vectorise(self):
        mouse_coordinates = self.data[:, 1:3].astype(np.float32)
        return mouse_coordinates[:-1] - mouse_coordinates[1:]

    def replace_mouse_event_sequence(self, deltas):
        data = self.data.copy()
        data[1:, 1] = np.round(data[0, 1] - np.cumsum(deltas[:, 0])).astype(data.dtype)
        data[1:, 2] = np.round(data[0, 2] - np.cumsum(deltas[:, 1])).astype(data.dtype)
        return MouseEventSequence(data)

def chunkify_dataset(dataset):
    chunks = []
    for mouse_event_sequence in dataset: chunks.extend(mouse_event_sequence.chunkify())
    return chunks

@functools.lru_cache(maxsize=None)
def load_minecraft_mouse_dynamics_dataset():
    dataset = []

    dataset_zip_path = os.path.join("datasets", "Minecraft-Mouse-Dynamics-Dataset", "10extracted.zip")
    with zipfile.ZipFile(dataset_zip_path) as zip_obj:
        for zip_file_obj in tqdm.tqdm(zip_obj.infolist()):
            if zip_file_obj.filename.endswith("_raw.csv"):
                with zip_obj.open(zip_file_obj) as csv_file:
                    dataset.append(MouseEventSequence(pd.read_csv(csv_file)))

    return chunkify_dataset(dataset)

def _load_mouse_dynamics_challenge_dataset(dataset_path):
    dataset = []

    dataset_path = os.path.join("datasets", "Mouse-Dynamics-Challenge", dataset_path)
    for file_name in tqdm.tqdm(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, file_name)

        for csv_file in os.listdir(subject_path):
            df = pd.read_csv(os.path.join(subject_path, csv_file), usecols=["client timestamp", "x", "y", "button"])
            df = df.rename(columns={"client timestamp": "Timestamp", "x": "X", "y": "Y"})
            df["Button Pressed"] = (df["button"] != "NoButton").astype(int)
            df["Subject ID"] = int(file_name.lstrip("user"))
            dataset.append(MouseEventSequence(df))

    return chunkify_dataset(dataset)

@functools.lru_cache(maxsize=None)
def load_mouse_dynamics_challenge_dataset():
    return (
        _load_mouse_dynamics_challenge_dataset("training_files"),
        _load_mouse_dynamics_challenge_dataset("test_files"),
    )

def load_amalgamated_mouse_dynamics_dataset():
    train, test = load_mouse_dynamics_challenge_dataset()
    minecraft_mouse_dynamics_dataset = load_minecraft_mouse_dynamics_dataset()

    amalgamated_mouse_dynamics_dataset = list(train) + list(test)
    minecraft_mouse_dynamics_dataset_offset = max([mouse_event_sequence.subject_id for mouse_event_sequence in amalgamated_mouse_dynamics_dataset]) + 1
    for mouse_event_sequence in minecraft_mouse_dynamics_dataset:
        mouse_event_sequence.subject_id += minecraft_mouse_dynamics_dataset_offset # Offsetting avoids label collisions

    return amalgamated_mouse_dynamics_dataset + minecraft_mouse_dynamics_dataset
