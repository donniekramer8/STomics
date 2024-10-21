#!/bin/bash
set -e

# conda activate istar
# cd /home/donald/Documents/istar/

prefix="/home/donald/Desktop/Andre_expansion/data/CODA Fallopian tube/visium hd/LK01JHU510_000_analysis/count/AJER376R/outs/binned_outputs/square_016um/donald_analysis/AJER376R_predictions/"

device="cuda"  # "cuda" or "cpu"

# visualize imputed gene expression
python impute.py "${prefix}" --load-saved --device=${device}  # train model from scratch

python plot_imputed.py "${prefix}"


# visualize spot-level gene expression data
# python plot_spots.py "${prefix}"
