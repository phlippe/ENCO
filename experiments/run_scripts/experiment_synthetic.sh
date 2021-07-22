#!/bin/sh

cd ../  # Go back to experiment direction
python run_exported_graphs.py --cluster \
                              --graph_files ../causal_graphs/synthetic_graphs/*.npz \
                              --seed 42