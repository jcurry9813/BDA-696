#!/usr/bin/env bash

source ../venv/bin/activate

pip install -r ../requirements.txt

rm -rf -- "plots"
mkdir -p -- "plots"

python3 A4.py

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open ./plots/response.html
elif [[ "$OSTYPE" == "darwin"* ]]; then
        open ./plots/response.html
fi
