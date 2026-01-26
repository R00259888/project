#!/bin/sh

if [ ! -d "datasets/IKDD" ]; then
    git clone https://github.com/MachineLearningVisionRG/IKDD datasets/IKDD
fi
if [ ! -d "datasets/Minecraft-Mouse-Dynamics-Dataset" ]; then
    git clone https://github.com/NyleSiddiqui/Minecraft-Mouse-Dynamics-Dataset datasets/Minecraft-Mouse-Dynamics-Dataset
fi

python3 -m pip install -r requirements.txt

python3 -m src.main --model keystroke --subject_id 1
python3 -m src.main --model mouse --subject_id 0

(
    cd thesis
    latexmk -pdf Thesis.tex
    open Thesis.pdf
)
