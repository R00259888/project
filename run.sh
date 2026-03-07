#!/bin/sh

mkdir -p datasets
if [ ! -d "datasets/IKDD" ]; then
    git clone https://github.com/MachineLearningVisionRG/IKDD datasets/IKDD
fi
if [ ! -d "datasets/Minecraft-Mouse-Dynamics-Dataset" ]; then
    git clone https://github.com/NyleSiddiqui/Minecraft-Mouse-Dynamics-Dataset datasets/Minecraft-Mouse-Dynamics-Dataset
fi
if [ ! -d "datasets/Mouse-Dynamics-Challenge" ]; then
    git clone https://github.com/balabit/Mouse-Dynamics-Challenge datasets/Mouse-Dynamics-Challenge
fi
if [ ! -d "datasets/KeyRecs" ]; then
    mkdir -p datasets/KeyRecs
    curl https://zenodo.org/records/7886743/files/fixed-text.csv -o datasets/KeyRecs/fixed-text.csv
    curl https://zenodo.org/records/7886743/files/free-text.csv -o datasets/KeyRecs/free-text.csv
fi
if [ ! -d "datasets/KeystrokeDynamicsBenchmarkDataset" ]; then
    mkdir -p datasets/KeystrokeDynamicsBenchmarkDataset
    curl https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv -o datasets/KeystrokeDynamicsBenchmarkDataset/DSL-StrongPasswordData.csv
fi

if [ -z "$COLAB_MODE" ]; then
    python3 -m pip install -r requirements.txt
fi

python3 -m src.experiments

if [ -z "$COLAB_MODE" ]; then
    (
        cd thesis
        latexmk -pdf Thesis.tex
        open Thesis.pdf
    )
fi
