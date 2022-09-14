#!/bin/bash

yes | conda create --name chromo python=3.9.12
eval "$(conda shell.bash hook)"
conda activate chromo
yes | pip install -r requirements.txt
pip install -e .
