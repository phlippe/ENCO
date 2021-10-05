#!/bin/sh

cd ../  # Go back to experiment direction
python run_exported_graphs.py --graph_files ../causal_graphs/synthetic_graphs/*.npz \
                              --weight_decay 1e-4 \
                              --seed 42