import argparse, random

from .dataset import load_mouse_dynamics_dataset, load_keystroke_dynamics_dataset
from .models.keystroke_dynamics_nn import KeystrokeDynamicsNNModel
from .models.mouse_dynamics_lstm import MouseDynamicsLSTMModel

def get_model(model):
    match model:
        case "keystroke":
            return KeystrokeDynamicsNNModel(load_keystroke_dynamics_dataset())
        case "mouse":
            return MouseDynamicsLSTMModel(load_mouse_dynamics_dataset())

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--epochs", type=int, default=10)
    argument_parser.add_argument("--model", choices=["keystroke", "mouse"], required=True)
    argument_parser.add_argument("--seed", type=int, default=0)

    args = argument_parser.parse_args()
    random.seed(args.seed)
    model = get_model(args.model)
    model.fit(args.epochs)

if __name__ == "__main__":
    main()
