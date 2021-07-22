"""
Functions for generating a graph and exporting its samples into a numpy array.
Used to setup benchmark datasets across all methods including baselines.
"""
import os
import numpy as np
import sys
sys.path.append("../")

from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.graph_definition import CausalDAGDataset
from causal_graphs.graph_generation import get_graph_func, generate_categorical_graph


def export_graph(filename, graph, num_obs, num_int):
    """
    Takes a graph and samples 'num_obs' observational data points and 'num_int' interventional data points
    per variable. All those are saved in the file 'filename'

    Parameters
    ----------
    filename : str
               Filename to save the exported graph to.
    graph : CausalDAG
            Causal graph to sample from and export.
    num_obs : int
              Number of observational data points to sample.
    num_int : int
              Number of data points to sample per intervention.
    """
    # Sample observational dataset
    data_obs = graph.sample(batch_size=num_obs, as_array=True)
    # Sample interventional dataset
    data_int = []
    for var_idx in range(graph.num_latents, graph.num_vars):
        var = graph.variables[var_idx]
        values = np.random.randint(var.prob_dist.num_categs, size=(num_int,))
        int_sample = graph.sample(interventions={var.name: values},
                                  batch_size=num_int,
                                  as_array=True)
        data_int.append(int_sample)
    # Stack all data
    data_int = np.stack(data_int, axis=0)
    data_obs = data_obs.astype(np.uint8)
    data_int = data_int.astype(np.uint8)
    adj_matrix = graph.adj_matrix
    # If the graph has latent variable, remove them from the dataset
    latents = graph.latents
    if graph.num_latents > 0:
        data_obs = data_obs[:, graph.num_latents:]
        data_int = data_int[:, :, graph.num_latents:]
        adj_matrix = adj_matrix[graph.num_latents:, graph.num_latents:]
        latents = latents - graph.num_latents  # Correcting indices
    # Export and visualize
    np.savez_compressed(filename, data_obs=data_obs, data_int=data_int,
                        adj_matrix=adj_matrix,
                        latents=latents)
    if graph.num_vars <= 100:
        for i, v in enumerate(graph.variables):
            v.name = r"$X_{%i}$" % (i+1)
        visualize_graph(graph,
                        filename=filename+".pdf",
                        figsize=(8, 8),
                        layout="graphviz")


def process_graphs(args):
    """
    Takes input arguments from the parser below, and creates and exports corresponding graphs.

    Parameters
    ----------
    args : Namespace
           Parsed arguments from the argument parser below.
    """
    os.makedirs(args.output_folder, exist_ok=True)

    for graph_type in args.graph_type:
        for graph_idx in range(args.num_graphs):
            seed = args.seed+graph_idx
            graph = create_graph(num_vars=args.num_vars,
                                 num_categs=args.num_categs,
                                 edge_prob=args.edge_prob,
                                 graph_type=graph_type,
                                 num_latents=args.num_latents,
                                 deterministic=args.deterministic,
                                 seed=seed)
            name = 'graph_%s_%i_%i' % (graph_type, args.num_vars, seed)
            if args.num_latents > 0:
                name += '_l%i' % (args.num_latents)
            export_graph(filename=os.path.join(args.output_folder, name),
                         graph=graph,
                         num_obs=args.num_obs,
                         num_int=args.num_int)


def create_graph(num_vars, num_categs, edge_prob, graph_type, num_latents, deterministic, seed):
    """
    Function for simplifying graph generation. See 'generate_categorical_graph' for argument details.
    """
    graph = generate_categorical_graph(num_vars=num_vars,
                                       min_categs=num_categs,
                                       max_categs=num_categs,
                                       edge_prob=edge_prob,
                                       connected=True,
                                       use_nn=True,
                                       deterministic=deterministic,
                                       graph_func=get_graph_func(graph_type),
                                       num_latents=num_latents,
                                       seed=seed)
    return graph


def load_graph(filename):
    """
    Function for loading an export graph again. Used in experiments.

    Parameters
    ----------
    filename : str
               Path of the file that should be loaded.
    """
    arr = np.load(filename)
    graph = CausalDAGDataset(adj_matrix=arr["adj_matrix"],
                             data_obs=arr["data_obs"].astype(np.int32),
                             data_int=arr["data_int"].astype(np.int32),
                             latents=arr["latents"] if "latents" in arr else None)
    return graph


if __name__ == '__main__':
    """
    Run this file to generate and export any graphs from the paper.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to save exported graphs to.')
    parser.add_argument('--graph_type', type=str, nargs='+', required=True,
                        help='Graph structure types to generate. See get_graph_func for details.')
    parser.add_argument('--num_graphs', type=int, default=1,
                        help='Number of graphs to generate and export.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for generating the graphs. If more than one graph is generated, '
                             'the seed is incremented along with the graph count.')
    parser.add_argument('--num_vars', type=int, default=25,
                        help='Number of variables that the graphs should have.')
    parser.add_argument('--num_obs', type=int, default=100000,
                        help='Number of samples to use for the observational dataset.')
    parser.add_argument('--num_int', type=int, default=10000,
                        help='Number of samples to use for the interventional dataset per variable.')
    parser.add_argument('--num_categs', type=int, default=10,
                        help='Number of categories/values that each variable can has.')
    parser.add_argument('--edge_prob', type=float, default=0.5,
                        help='For random graph structure, with which probability to connect two variables.')
    parser.add_argument('--num_latents', type=int, default=0,
                        help='Number of latent confounders to add to the graph. Requires random graph structures.')
    parser.add_argument('--deterministic', action='store_true',
                        help='If True, all probability distributions become deterministic. Otherwise, we use '
                             'soft distributions with all values having a probability greater zero.')

    args = parser.parse_args()
    assert args.num_latents == 0 or args.graph_type == ["random"], \
        "For latent variables, you need to select \"random\" as graph type."
    process_graphs(args)
