#!/bin/sh

cd ../  # Go back to experiment direction
# Change the value of 'max_inters' to the number of variables to use interventional data for
python run_exported_graphs.py --graph_files ../causal_graphs/synthetic_graphs_partial/*_42.npz \
                              --weight_decay 4e-5 \
                              --max_inters 10 \
                              --lambda_sparse 0.002 \
                              --seed 42