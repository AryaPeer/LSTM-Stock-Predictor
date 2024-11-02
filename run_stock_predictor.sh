#!/bin/bash

ENV_NAME="lstm_env"
MAIN_DIR="main"

if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $ENV_NAME
fi

source $ENV_NAME/bin/activate  

pip install --upgrade pip

if [ -f requirements.txt ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Please provide a requirements file."
    exit 1
fi

cd $MAIN_DIR
nohup python main.py >/dev/null 2>&1 &