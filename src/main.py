import argparse, collections, functools, random

import bob.measure
import numpy as np
import tensorflow as tf

from .attacks import impersonation_attack
from .defences import data_augmentation_defence
from .dataset import load_minecraft_mouse_dynamics_dataset, load_ikdd_keystroke_dynamics_dataset, load_mouse_dynamics_challenge_dataset
from .models.keystroke_dynamics_nn import KeystrokeDynamicsNNModel
from .models.mouse_dynamics_lstm import MouseDynamicsLSTMModel
from .models.mouse_dynamics_cnn_and_lstm import MouseDynamicsCNNAndLSTMModel

@functools.lru_cache(maxsize=None)
def get_dataset(model):
    match model:
        case "IKDD":
            return load_ikdd_keystroke_dynamics_dataset()
        case "Minecraft-Mouse-Dynamics-Dataset":
            return load_minecraft_mouse_dynamics_dataset()
        case "Mouse-Dynamics-Challenge":
            return load_mouse_dynamics_challenge_dataset()

def __train_test_split(dataset, train_perc=0.7):
    subjects = collections.defaultdict(list)
    for sequence in dataset:
        subjects[sequence.subject_id].append(sequence)

    train_dataset, test_dataset = [], []
    for _, sequences in subjects.items():
        dataset_split_index = int(len(sequences) * train_perc)
        if dataset_split_index == 0:
            train_dataset.extend(sequences)
            test_dataset.extend(sequences)
        else:
            train_dataset.extend(sequences[:dataset_split_index])
            test_dataset.extend(sequences[dataset_split_index:])

    return train_dataset, test_dataset

def train_test_split(dataset):
    if type(dataset) == tuple: return dataset # Already split
    else: return __train_test_split(dataset)

def get_subject_ids(dataset):
    if type(dataset) == tuple: dataset = dataset[0]
    return set([sequence.subject_id for sequence in dataset])

def get_model(model, dataset, subject_id):
    match model:
        case "KeystrokeDynamicsNNModel":
            return KeystrokeDynamicsNNModel(dataset, subject_id)
        case "MouseDynamicsLSTMModel":
            return MouseDynamicsLSTMModel(dataset, subject_id)
        case "MouseDynamicsCNNAndLSTMModel":
            return MouseDynamicsCNNAndLSTMModel(dataset, subject_id)

def get_metrics(model, test_dataset, subject_id, attack):
    if attack == "impersonation": test_dataset = impersonation_attack(test_dataset, subject_id)
    X, y_desired = model.prepare_features(test_dataset)
    confidence_score = model.predict(X).flatten()

    negatives = confidence_score[np.array(y_desired) == 0]
    positives = confidence_score[np.array(y_desired) == 1]

    if len(negatives) == 0 or len(positives) == 0 or np.isnan(confidence_score).any():
        return {"eer": float("nan"), "far": float("nan"), "frr": float("nan")}
    eer_threshold = bob.measure.eer_threshold(negatives, positives)
    far, frr = bob.measure.farfrr(negatives, positives, eer_threshold)
    eer = (far + frr) / 2
    return {"eer": eer, "far": far, "frr": frr}

def __compute_class_weight(train, subject_id):
    positive_sum = sum([1 for sequence in train if sequence.subject_id == subject_id])
    negative_sum = len(train) - positive_sum

    if positive_sum > 0: return {0: 1.0, 1: negative_sum / positive_sum}
    return {0: 1.0, 1: 1.0}

def train_model(model, subject_id, train, defence, epochs, seed):
    set_random_seed(seed)
    train = list(train)

    if defence == "augmentation": train = data_augmentation_defence(train, subject_id)

    class_weight = __compute_class_weight(train, subject_id)

    model = get_model(model, train, subject_id)
    model.fit(epochs, class_weight=class_weight)
    return model

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--attack", choices=["None", "impersonation"], required=False)
    argument_parser.add_argument("--defence", choices=["None", "augmentation"], required=False)
    argument_parser.add_argument("--epochs", type=int, default=10)
    argument_parser.add_argument("--evaluation_plot", type=str, required=False)
    argument_parser.add_argument("--dataset", choices=["IKDD", "Minecraft-Mouse-Dynamics-Dataset", "Mouse-Dynamics-Challenge"], required=True)
    argument_parser.add_argument("--model", choices=["KeystrokeDynamicsNNModel", "MouseDynamicsLSTMModel", "MouseDynamicsCNNAndLSTMModel"], required=True)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--subject_id", type=int, required=True)

    args = argument_parser.parse_args()

    dataset = get_dataset(args.dataset)
    train, test = train_test_split(dataset)

    model = train_model(args.model, args.subject_id, train, args.defence, args.epochs, args.seed)
    print(get_metrics(model, test, args.subject_id, args.attack))

if __name__ == "__main__":
    main()
