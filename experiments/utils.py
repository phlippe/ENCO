import torch
import numpy as np
import time
from argparse import ArgumentParser
from copy import deepcopy
import json
import random
import os
import sys
sys.path.append("../")

from causal_graphs.graph_utils import adj_matrix_to_edges
from causal_graphs.graph_visualization import visualize_graph
from causal_discovery.utils import get_device
from causal_discovery.enco import ENCO

def set_seed(seed):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_basic_parser():
    """
    Returns argument parser of standard hyperparameters/experiment arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs to run ENCO for.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the experiments.')
    parser.add_argument('--cluster', action='store_true',
                        help='If True, no tqdm progress bars are used.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use for distribution and graph fitting.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of the distribution fitting NNs.')
    parser.add_argument('--model_iters', type=int, default=1000,
                        help='Number of updates per distribution fitting stage.')
    parser.add_argument('--graph_iters', type=int, default=100,
                        help='Number of updates per graph fitting stage.')
    parser.add_argument('--lambda_sparse', type=float, default=0.004,
                        help='Sparsity regularizer in the graph fitting stage.')
    parser.add_argument('--lr_model', type=float, default=5e-3,
                        help='Learning rate of distribution fitting NNs.')
    parser.add_argument('--lr_gamma', type=float, default=2e-2,
                        help='Learning rate of gamma parameters in graph fitting.')
    parser.add_argument('--lr_theta', type=float, default=1e-1,
                        help='Learning rate of theta parameters in graph fitting.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save experiment log to. If None, one will'
                             ' be created based on the current time')
    parser.add_argument('--GF_num_batches', type=int, default=1,
                        help='Number of batches to use in graph fitting gradient estimators.')
    parser.add_argument('--GF_num_graphs', type=int, default=100,
                        help='Number of graph samples to use in the gradient estimators.')
    parser.add_argument('--max_graph_stacking', type=int, default=200,
                        help='Number of graphs to evaluate in parallel. Reduce this to save memory.')
    parser.add_argument('--use_theta_only_stage', action='store_true',
                        help='If True, gamma is frozen in every second graph fitting stage.'
                             ' Recommended for large graphs with >=100 nodes.')
    parser.add_argument('--theta_only_num_graphs', type=int, default=4,
                        help='Number of graph samples to use when gamma is frozen.')
    parser.add_argument('--theta_only_iters', type=int, default=1000,
                        help='Number of updates per graph fitting stage when gamma is frozen.')
    parser.add_argument('--save_model', action='store_true',
                        help='If True, the neural networks will be saved besides gamma and theta.')
    parser.add_argument('--stop_early', action='store_true',
                        help='If True, ENCO stops running if it achieved perfect reconstruction in'
                             ' all of the last 5 epochs.')
    return parser


def test_graph(graph, args, checkpoint_dir, file_id):
    """
    Runs ENCO on a given graph for structure learning.

    Parameters
    ----------
    graph : CausalDAG
            The graph on which we want to perform causal structure learning.
    args : Namespace
           Parsed input arguments from the argument parser, including all
           hyperparameters.
    checkpoint_dir : str
                     Directory to which all logs and the model should be
                     saved to.
    file_id : str
              Identifier of the graph/experiment instance. Is used for creating
              log filenames, and identify the graph among other experiments in
              the same checkpoint directory.
    """
    # Execute ENCO on graph
    discovery_module = ENCO(graph=graph,
                            hidden_dims=[args.hidden_size],
                            lr_model=args.lr_model,
                            lr_gamma=args.lr_gamma,
                            lr_theta=args.lr_theta,
                            model_iters=args.model_iters,
                            graph_iters=args.graph_iters,
                            batch_size=args.batch_size,
                            GF_num_batches=args.GF_num_batches,
                            GF_num_graphs=args.GF_num_graphs,
                            lambda_sparse=args.lambda_sparse,
                            use_theta_only_stage=args.use_theta_only_stage,
                            theta_only_num_graphs=args.theta_only_num_graphs,
                            theta_only_iters=args.theta_only_iters,
                            max_graph_stacking=args.max_graph_stacking,
                            )
    discovery_module.to(get_device())
    start_time = time.time()
    discovery_module.discover_graph(num_epochs=args.num_epochs,
                                    stop_early=args.stop_early)
    duration = int(time.time() - start_time)
    print("-> Finished training in %ih %imin %is" % (duration // 3600, (duration // 60) % 60, duration % 60))

    # Save metrics in checkpoint folder
    metrics = discovery_module.get_metrics()
    with open(os.path.join(checkpoint_dir, "metrics_%s.json" % file_id), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(checkpoint_dir, "metrics_full_log_%s.json" % file_id), "w") as f:
        json.dump(discovery_module.metric_log, f, indent=4)

    # Save predicted binary matrix
    binary_matrix = discovery_module.get_binary_adjmatrix().detach().cpu().numpy()
    np.save(os.path.join(checkpoint_dir, 'binary_matrix_%s.npy' % file_id),
            binary_matrix.astype(np.bool))

    # Visualize predicted graphs. For large graphs, visualizing them do not really help
    if graph.num_vars < 40:
        pred_graph = deepcopy(graph)
        pred_graph.adj_matrix = binary_matrix
        pred_graph.edges = adj_matrix_to_edges(pred_graph.adj_matrix)
        figsize = max(3, pred_graph.num_vars / 1.5)
        visualize_graph(pred_graph,
                        filename=os.path.join(checkpoint_dir, "graph_%s_prediction.pdf" % (file_id)),
                        figsize=(figsize, figsize),
                        layout="circular")

    # Save parameters and model if wanted
    state_dict = discovery_module.get_state_dict()
    if not args.save_model:
        _ = state_dict.pop("model")
    torch.save(state_dict,
               os.path.join(checkpoint_dir, "state_dict_%s.tar" % file_id))
