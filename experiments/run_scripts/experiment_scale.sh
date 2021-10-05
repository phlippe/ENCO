#!/bin/sh
# Graphs are dynamically generated here to save disk memory

cd ../  # Go back to experiment direction
# Graph with 100 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 100 \
                               --edge_prob 0.08 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 50 \
                               --model_iters 2000 \
                               --use_theta_only_stage \
                               --theta_only_iters 1000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed 42
# Graph with 200 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 200 \
                               --edge_prob 0.04 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 20 \
                               --model_iters 2000 \
                               --use_theta_only_stage \
                               --theta_only_iters 1000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed 42
# Graph with 400 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 400 \
                               --edge_prob 0.02 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 10 \
                               --model_iters 4000 \
                               --use_theta_only_stage \
                               --theta_only_iters 2000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed 42
# Graph with 1000 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 1000 \
                               --edge_prob 0.008 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 2 \
                               --batch_size 64 \
                               --GF_num_batches 2 \
                               --model_iters 4000 \
                               --use_theta_only_stage \
                               --theta_only_iters 2000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed 42