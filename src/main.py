import argparse, random

import numpy as np
import tensorflow as tf

from .dataset import load_mouse_dynamics_dataset, load_keystroke_dynamics_dataset
from .models.keystroke_dynamics_nn import KeystrokeDynamicsNNModel
from .models.mouse_dynamics_lstm import MouseDynamicsLSTMModel

def get_dataset(model):
    match model:
        case "keystroke":
            return load_keystroke_dynamics_dataset()
        case "mouse":
            return load_mouse_dynamics_dataset()

def train_test_split(dataset, train_perc=0.7):
    dataset, subject_ids = list(dataset), [sequence.subject_id for sequence in dataset]
    dataset_evenly_distributed = []

    while len(dataset):
        for subject_id in range(min(subject_ids), max(subject_ids) + 1):
            for i in range(len(dataset)):
                if dataset[i].subject_id == subject_id:
                    dataset_evenly_distributed.append(dataset.pop(i))
                    break

    dataset_split_index = int(len(dataset_evenly_distributed) * train_perc)
    train_dataset = dataset_evenly_distributed[:dataset_split_index]
    test_dataset = dataset_evenly_distributed[dataset_split_index:]
    return train_dataset, test_dataset

def get_model(model, dataset, subject_id):
    match model:
        case "keystroke":
            return KeystrokeDynamicsNNModel(dataset, subject_id)
        case "mouse":
            return MouseDynamicsLSTMModel(dataset, subject_id)

def evaluate_model(model, test_dataset):
    X, y_desired = model.prepare_features(test_dataset)
    y = model.predict(X).flatten() >= 0.5 # Convert confidence to boolean.
    mean_accuracy = np.mean(y == y_desired)
    print("Accuracy:", mean_accuracy)

def set_random_seed(seed):
    random.seed(seed)
    tf.random.set_seed(seed)

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--epochs", type=int, default=10)
    argument_parser.add_argument("--model", choices=["keystroke", "mouse"], required=True)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--subject_id", type=int, required=True)

    args = argument_parser.parse_args()
    set_random_seed(args.seed)

    dataset = get_dataset(args.model)
    train, test = train_test_split(dataset)

    model = get_model(args.model, train, args.subject_id)
    model.fit(args.epochs)

    evaluate_model(model, test)

if __name__ == "__main__":
    main()
