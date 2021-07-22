"""
Utilities for generating different graph structures. 
The code currently supports "random", "chain", "bidiag", "collider", "jungle", "full", and "regular". 
The function 'generate_categorical_graph' combines the generation
of the adjacency matrix and conditional distributions.
"""
import torch
import numpy as np
from random import shuffle
import networkx as nx
import random
import string
import sys
sys.path.append("../")

from causal_graphs.graph_definition import CausalDAG, CausalVariable
from causal_graphs.variable_distributions import get_random_categorical
from causal_graphs.graph_utils import edges_to_adj_matrix


def graph_from_adjmatrix(variable_names, dist_func, adj_matrix, latents=None):
    """
    Creates a CausalDAG object from an adjacency matrix.

    Parameters
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    adj_matrix : np.ndarray, shape [num_vars, num_vars], type np.bool
                 Numpy array representing the adjacency matrix between the variable. An entry (i,j)
                 being true represents the edge X_i->X_j.
    latents : None / np.ndarray, shape [num_latents, 3], type np.int32
              Numpy array representing latent variables. For each row, the first entry is the variable
              index of the latent confounder, and the other two the children of the confounder. If no
              latent confounders exists, use "None" as input.
    """
    variables = []
    for v_idx, name in enumerate(variable_names):
        parents = np.where(adj_matrix[:, v_idx])[0]
        prob_dist = dist_func(input_names=[variable_names[p] for p in parents], name=name)
        var = CausalVariable(name=name, prob_dist=prob_dist)
        variables.append(var)

    graph = CausalDAG(variables, adj_matrix=adj_matrix, latents=latents)
    return graph


def graph_from_edges(variable_names, dist_func, edges, latents=None):
    """
    Same as graph_from_adjmatrix, just with edges instead of an adjacency matrix.
    """
    adj_matrix = edges_to_adj_matrix(edges, len(variable_names))
    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix, latents=latents)


def generate_random_graph(variable_names, dist_func, edge_prob, connected=False, max_parents=-1, num_latents=0, **kwargs):
    """
    Generates a graph structure which follows the 'random' graph structure. In this,
    nodes are randomly connected.

    Parameters 
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    edge_prob : float
                Probability that two arbitrary nodes have an edge between each other.
    connected : bool
                If True, it is ensured that the graph is one, connected graph independent of the edge
                probability. This means that ignoring the edge directions, every node can be reached
                from any other. 
    max_parents : int
                  Maximum number of parents that a node is allowed to have. If the argument is negative,
                  the maximum is unlimited. 
    num_latents : int
                  Number of latent confounders that should randomly be added to the graph. 
    """
    shuffle(variable_names)  # To have a random order
    num_vars = len(variable_names)

    # Generate random adjacency matrix with specified edge probability
    adj_matrix = np.random.binomial(n=1, p=edge_prob, size=(num_vars, num_vars))

    # Make sure that adjacency matrix is half diagonal
    for v_idx in range(num_vars):
        adj_matrix[v_idx, :v_idx+1] = 0

    # Nodes that do not have any parents or children are connected
    for v_idx in range(num_vars):
        has_connection = (adj_matrix[v_idx, :].any() or adj_matrix[:, v_idx].any())
        if not has_connection:
            con_idx = np.random.randint(num_vars-1)
            if con_idx >= v_idx:
                con_idx += 1
                adj_matrix[v_idx, con_idx] = True
            else:
                adj_matrix[con_idx, v_idx] = True

    # Ensure that a node has less than N parents
    if max_parents > 0:
        for v_idx in range(adj_matrix.shape[0]):
            num_parents = adj_matrix[:, v_idx].sum()
            if num_parents > max_parents:
                indices = np.where(adj_matrix[:, v_idx] == 1)[0]
                indices = indices[np.random.permutation(indices.shape[0])[:num_parents-max_parents]]
                adj_matrix[indices, v_idx] = 0

    # Connect nodes to one connected graph
    if connected:
        visited_nodes, connected_nodes = [], [0]
        while len(visited_nodes) < num_vars:
            while len(connected_nodes) > 0:
                v_idx = connected_nodes.pop(0)
                children = np.where(adj_matrix[v_idx, :])[0].tolist()
                parents = np.where(adj_matrix[:, v_idx])[0].tolist()
                neighbours = children + parents
                for n in neighbours:
                    if (n not in visited_nodes) and (n not in connected_nodes):
                        connected_nodes.append(n)
                if v_idx not in visited_nodes:
                    visited_nodes.append(v_idx)
            if len(visited_nodes) < num_vars:
                node1 = np.random.choice(np.array(visited_nodes))
                node2 = np.random.choice(np.array([i for i in range(num_vars) if i not in visited_nodes]))
                adj_matrix[min(node1, node2), max(node1, node2)] = True
                connected_nodes.append(node1)

    # Add latent confounders 
    if num_latents > 0:
        # Latent confounders are identified by their variable name "X_{l,...}"
        variable_names = [r"$X_{l,%i}$" % (i+1) for i in range(num_latents)] + variable_names
        # Latent confounders are added in the graph structure. When exporting the graph, 
        # we remove those variables so that we can apply our structure learning algorithm
        # without any changes.
        node_idxs = [v_idx+num_latents for v_idx in range(num_vars)
                     if (adj_matrix[:, v_idx].sum() < max_parents or max_parents <= 0)]
        adj_matrix = np.concatenate([np.zeros((num_latents, num_vars)), adj_matrix], axis=0)
        adj_matrix = np.concatenate([np.zeros((num_vars+num_latents, num_latents)), adj_matrix], axis=1)
        # Randomly select the node pairs on which we want to have a latent confounder
        latent_children = []
        for l in range(num_latents):
            node_pair = None
            # We sample unique node pairs where there exists no direct edge between both nodes
            while node_pair is None or node_pair in latent_children or adj_matrix[node_pair[0], node_pair[1]]:
                node_pair = random.sample(node_idxs, k=2)
                node_pair = sorted(node_pair)
            latent_children.append(node_pair)
            adj_matrix[l, node_pair[0]] = 1
            adj_matrix[l, node_pair[1]] = 1
        latents = np.array([[i]+lc for i, lc in enumerate(latent_children)])
    else:
        latents = None

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix, latents=latents)


def generate_chain(variable_names, dist_func, **kwargs):
    """
    Generates a graph structure which follows the 'chain' graph structure. In this,
    each node is connected to one parent and one child, except the head and tail nodes.

    Parameters 
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    """
    shuffle(variable_names)  # To have a random order
    num_vars = len(variable_names)

    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    for v_idx in range(num_vars-1):
        adj_matrix[v_idx, v_idx+1] = True

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_bidiag(variable_names, dist_func, **kwargs):
    """
    Generates a graph structure which follows the 'bidiag' graph structure. It is a
    chain with additional connections from parents of parents and children of children.

    Parameters 
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    """
    shuffle(variable_names)
    num_vars = len(variable_names)

    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    for v_idx in range(num_vars-1):
        adj_matrix[v_idx, v_idx+1] = True
        if v_idx < num_vars - 2:
            adj_matrix[v_idx, v_idx+2] = True

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_collider(variable_names, dist_func, **kwargs):
    """
    Generates a graph structure which follows the 'collider' graph structure. One variable
    has all other variables as parents.

    Parameters 
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    """
    shuffle(variable_names)
    num_vars = len(variable_names)

    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    adj_matrix[:-1, -1] = True

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_jungle(variable_names, dist_func, num_levels=2, **kwargs):
    """
    Generates a graph structure which follows the 'jungle' graph structure. The graph 
    has a binary tree-like structure, where we allow additional connections from 
    parents of higher levels to children of lower levels.

    Parameters 
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    num_levels : int
                 A node is connected to all of its (indirect) children 'num_levels' lower levels.
                 num_levels=1 represents a standard binary tree, num_levels=2 additionally connects
                 node to its children of children, and so on. 
    """
    shuffle(variable_names)
    num_vars = len(variable_names)

    edges = []
    for i in range(num_vars):
        level = int(np.log2(i+1))
        idx = i + 1 - 2 ** level
        for l in range(1, num_levels+1):
            gl = (2**l) * idx + 2 ** (level + l) - 1
            edges += [[i, gl + j] for j in range(2**l)]
    edges = [e for e in edges if max(e) < num_vars]

    return graph_from_edges(variable_names, dist_func, edges)


def generate_full(variable_names, dist_func, **kwargs):
    """
    Generates a graph structure which follows the 'full' graph structure. It is the same
    as a random graph with edge probability 1.0.

    Parameters 
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    """
    return generate_random_graph(variable_names, dist_func, edge_prob=1.0)


def generate_regular_graph(variable_names, dist_func, num_neigh=10, **kwargs):
    """
    Generates a graph structure which follows the 'regular' graph structure. Thereby,
    every node is connected to num_neigh neighbours.

    Parameters 
    ----------
    variable_names : list[str]
                     List of names that the variables should have. Used for visualization and
                     internal communication between variables. No two variables should have
                     the same name.
    dist_func : (list[str],name) -> ProbDist
                Function for generating the probability distribution. It should take as input a 
                list of the variables names representing the parents, and the name of the variable
                to model. As output, it is expected to give a ProbDist object.
    num_neigh : int
                Number of neighbours each node should have. Must be less than num_vars-1
    """
    shuffle(variable_names)
    num_vars = len(variable_names)
    num_neigh = min(num_neigh, num_vars-1)
    graphs = nx.random_graphs.random_regular_graph(num_neigh, num_vars)
    edges = np.array(graphs.edges())
    edges.sort(axis=-1)

    return graph_from_edges(variable_names, dist_func, edges)


def get_graph_func(name):
    """
    Converts a string describing the wanted graph structure into the corresponding function. 
    """
    if name == "chain":
        f = generate_chain
    elif name == "bidiag":
        f = generate_bidiag
    elif name == "collider":
        f = generate_collider
    elif name == "jungle":
        f = generate_jungle
    elif name == "full":
        f = generate_full
    elif name == "regular":
        f = generate_regular_graph
    elif name == "random":
        f = generate_random_graph
    elif name.startswith("random_max_"):  # Random graph with maximum number of parents
        max_parents = int(name.split("_")[-1])
        f = lambda *args, **kwargs: generate_random_graph(*args, max_parents=max_parents, **kwargs)
    else:
        f = generate_random_graph
    return f


def generate_categorical_graph(num_vars,
                               min_categs,
                               max_categs,
                               inputs_independent=False,
                               use_nn=True,
                               deterministic=False,
                               graph_func=generate_random_graph,
                               seed=-1,
                               **kwargs):
    """
    Summarizes the whole generation process for a graph with categorical variables. Returns a CausalDAG object.

    Parameters
    ----------
    num_vars : int
               Number of variables to have in the graph.
    min_categs : int
                 The number of categories per variable are randomly sampled between min_categs and max_categs.
                 If all variables should have the same number of categories, choose min_categs=max_categs.
    max_categs : int
                 See min_categs
    inputs_independent : bool
                         If True, defines the probability distribution as normalized product of indepedent
                         conditionals. For example, p(A|B,C)=p(A|B)*p(A|C)/sum_A p(A|B)*p(A|C).
    use_nn : bool
             If True, uses randomly initialized neural networks to model the conditional distributions. Default
             option for all expeirments in the paper.
    deterministic : bool
                    If True, uses random, but deterministic distributions for all variables.
    graph_func : function
                 One of the functions for generating the graph structure. Can be obtained from 'get_graph_func'.
    seed : int
           Seed to set for reproducible graph generation.
    kwargs : dict
             Any other argument that should be passed to the graph structure generating function.
    """
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    if num_vars <= 26:  # For less than 26 variables, we call the variables alphabetically, otherwise numerically
        variable_names = [n for i, n in zip(range(1, num_vars+1), string.ascii_uppercase)]
    else:
        variable_names = [r"$X_{%s}$" % i for i in range(1, num_vars+1)]
    var_num_categs = np.random.randint(min_categs, max_categs+1, size=(num_vars,))

    def dist_func(input_names, name):
        if min_categs != max_categs:
            input_num_categs = [var_num_categs[variable_names.index(v_name)] for v_name in input_names]
            num_categs = var_num_categs[variable_names.index(name)]
        else:
            input_num_categs, num_categs = [min_categs]*len(input_names), min_categs
        dist = get_random_categorical(input_names=input_names,
                                      input_num_categs=input_num_categs,
                                      num_categs=num_categs,
                                      inputs_independent=inputs_independent,
                                      use_nn=use_nn,
                                      deterministic=deterministic)
        return dist

    return graph_func(variable_names, dist_func, **kwargs)
