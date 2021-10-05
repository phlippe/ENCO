#!/bin/sh

cd ../  # Go back to experiment direction
python run_exported_graphs.py --graph_files ../causal_graphs/confounder_graphs/*.npz \
                              --sample_size_obs 5000 \
                              --sample_size_inters 512 \
                              --weight_decay 1e-4 \
                              --seed 42