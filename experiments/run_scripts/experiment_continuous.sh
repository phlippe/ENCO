#!/bin/sh

cd ../  # Go back to experiment direction
python run_exported_graphs.py --graph_files ../causal_graphs/continuous_graphs/*.npz \
                              --weight_decay 1e-4 \
                              --hidden_size 32 \
                              --lambda_sparse 0.001 \
                              --sample_size_obs 909 \
                              --sample_size_inters 909 \
                              --seed 42