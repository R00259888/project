import dataclasses, os, zipfile

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
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.mouse_event_sequence = [
                MouseEvent(
                    timestamp=int(mouse_event["Timestamp"]),
                    x=int(mouse_event["X"]),
                    y=int(mouse_event["Y"]),
                    button_pressed=int(mouse_event["Button Pressed"]),
                    subject_id=int(mouse_event["Subject ID"])
                )
                for _, mouse_event in data.iterrows()
            ]
        else:
            self.mouse_event_sequence = data
        self.subject_id = self.mouse_event_sequence[0].subject_id

    def __len__(self):
        return len(self.mouse_event_sequence)

    def __getitem__(self, key):
        value = self.mouse_event_sequence[key]
        if isinstance(key, slice): return MouseEventSequence(value)
        return value

    def chunkify(self, chunk_size=8192):
        return [self[i:i + chunk_size] for i in range(0, len(self), chunk_size)]

    def vectorise(self):
        mouse_coordinate_deltas = []
        for i in range(1, len(self.mouse_event_sequence)):
            mouse_coordinate_deltas.append([
                self.mouse_event_sequence[i - 1].x - self.mouse_event_sequence[i].x,
                self.mouse_event_sequence[i - 1].y - self.mouse_event_sequence[i].y
            ])
        return np.array(mouse_coordinate_deltas, dtype=np.float32)

def chunkify_dataset(dataset):
    chunks = []
    for mouse_event_sequence in dataset: chunks.extend(mouse_event_sequence.chunkify())
    return chunks

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
            df = pd.read_csv(os.path.join(subject_path, csv_file))
            df = df.rename(columns={"client timestamp": "Timestamp", "x": "X", "y": "Y"})
            df["Button Pressed"] = (df["button"] != "NoButton").astype(int)
            df["Subject ID"] = int(file_name.lstrip("user"))
            dataset.append(MouseEventSequence(df))

    return chunkify_dataset(dataset)

def load_mouse_dynamics_challenge_dataset():
    return (
        _load_mouse_dynamics_challenge_dataset("training_files"),
        _load_mouse_dynamics_challenge_dataset("test_files"),
    )
