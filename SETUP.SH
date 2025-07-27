#!/bin/bash
conda create -p venv python==3.9 -y
conda activate venv/
pip install -r requirements.txt
