import os, zipfile

def load_mouse_dynamics_dataset():
    dataset = []
    dataset_zip_path = os.path.join("datasets", "Minecraft-Mouse-Dynamics-Dataset", "10extracted.zip")
    with zipfile.ZipFile(dataset_zip_path) as zip_obj:
        for zip_file_obj in zip_obj.infolist():
            if zip_file_obj.filename.endswith("_raw.csv"):
                with zip_obj.open(zip_file_obj) as csv_file:
                    dataset += [csv_file.read()]
    return dataset
