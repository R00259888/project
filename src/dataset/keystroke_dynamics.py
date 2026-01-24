import os

def load_keystroke_dynamics_dataset():
    dataset = []
    dataset_path = os.path.join("datasets", "IKDD", "IKDD")
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(dataset_path, file_name)) as file_obj:
                dataset += [file_obj.read()]
    return dataset
