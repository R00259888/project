import argparse, random

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model", choices=["keystroke", "mouse"])
    argument_parser.add_argument("--seed", type=int, default=0)

    args = argument_parser.parse_args()
    random.seed(args.seed)

if __name__ == "__main__":
    main()
