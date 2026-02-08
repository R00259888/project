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
