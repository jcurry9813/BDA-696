#!/usr/bin/env bash

source ../venv/bin/activate

pip install -r ../requirements.txt

rm -rf -- "plots"
mkdir -p -- "plots"

python3 midterm.py

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open ./plots/results.html
        xdg-open ./plots/bfresults.html
        xdg-open ./plots/pcresults.html
        if [ -e ./plots/cat_cat_matrix.html ]
          then
          xdg-open ./plots/cat_cat_matrix.html
        fi
        if [ -e ./plots/cont_cont_matrix.html ]
          then
          xdg-open ./plots/cont_cont_matrix.html
        fi
        if [ -e ./plots/cat_cont_matrix.html ]
          then
          xdg-open ./plots/cat_cont_matrix.html
        fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
        open ./plots/results.html
        open ./plots/bfresults.html
        open ./plots/pcresults.html
        if [ -e ./plots/cat_cat_matrix.html ]
          then
          open ./plots/cat_cat_matrix.html
        fi
        if [ -e ./plots/cont_cont_matrix.html ]
          then
          open ./plots/cont_cont_matrix.html
        fi
        if [ -e ./plots/cat_cont_matrix.html ]
          then
          open ./plots/cat_cont_matrix.html
        fi
fi