import argparse, collections, random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .dataset import load_mouse_dynamics_dataset, load_keystroke_dynamics_dataset
from .models.keystroke_dynamics_nn import KeystrokeDynamicsNNModel
from .models.mouse_dynamics_lstm import MouseDynamicsLSTMModel
from .models.mouse_dynamics_cnn_and_lstm import MouseDynamicsCNNAndLSTMModel

def get_dataset(model):
    match model:
        case "IKDD":
            return load_keystroke_dynamics_dataset()
        case "Minecraft-Mouse-Dynamics-Dataset":
            return load_mouse_dynamics_dataset()
        case "Mouse-Dynamics-Challenge":
            return None

def train_test_split(dataset, train_perc=0.7):
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

def get_model(model, dataset, subject_id):
    match model:
        case "KeystrokeDynamicsNNModel":
            return KeystrokeDynamicsNNModel(dataset, subject_id)
        case "MouseDynamicsLSTMModel":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            return MouseDynamicsLSTMModel(dataset, subject_id)
        case "MouseDynamicsCNNAndLSTMModel":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            return MouseDynamicsCNNAndLSTMModel(dataset, subject_id)

def evaluate_model(model, test_dataset, model_subject_id, evaluation_plot):
    subject_ids = set([sequence.subject_id for sequence in test_dataset])

    subject_confidences = {}
    for subject_id in subject_ids:
        sequence = [*filter(lambda sequence: sequence.subject_id == subject_id, test_dataset)]
        X, _ = model.prepare_features(sequence)
        subject_confidences[subject_id] = np.mean(model.predict(X).flatten())

    top_10_subject_ids = sorted(subject_confidences.items(), key=lambda confidence: confidence[1], reverse=True)[:10]    
    top_10_subject_ids = [subject_confidence_item[0] for subject_confidence_item in top_10_subject_ids]
    bar_x = [str(subject_id) for subject_id in top_10_subject_ids]
    bar_height = [subject_confidences[subject_id] for subject_id in top_10_subject_ids]
    bar_colours = [*map(lambda subject_id: ("red" if subject_id == model_subject_id else "blue"), top_10_subject_ids)]

    plt.bar(bar_x, bar_height, color=bar_colours)
    plt.xlabel("Subject ID")
    plt.ylabel("Confidence")
    plt.title("Top 10 confidence scores for each subject, trained to detect subject: " + str(model_subject_id))
    plt.savefig(evaluation_plot)
    plt.close()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--epochs", type=int, default=10)
    argument_parser.add_argument("--evaluation_plot", type=str, required=False)
    argument_parser.add_argument("--dataset", choices=["IKDD", "Minecraft-Mouse-Dynamics-Dataset"], required=True)
    argument_parser.add_argument("--model", choices=["KeystrokeDynamicsNNModel", "MouseDynamicsLSTMModel", "MouseDynamicsCNNAndLSTMModel"], required=True)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--subject_id", type=int, required=True)

    args = argument_parser.parse_args()
    set_random_seed(args.seed)

    dataset = get_dataset(args.dataset)
    train, test = train_test_split(dataset)

    model = get_model(args.model, train, args.subject_id)
    model.fit(args.epochs)

    if args.evaluation_plot:
        evaluate_model(model, test, args.subject_id, args.evaluation_plot)

if __name__ == "__main__":
    main()
