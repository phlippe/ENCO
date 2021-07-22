"""
Wrapper classes for causal graphs. A causal graph is defined 
by a set of "CausalVariable" objects, each having a name and 
conditional probability distribution. Additionally, we have
an explicit representation of the adjacency matrix for easier
handling. 
"""
import torch
import numpy as np
from copy import deepcopy
import importlib
import sys
sys.path.append("../")

from causal_graphs.graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations
from causal_graphs.variable_distributions import ProbDist, ConstantDist, CategoricalDist, DiscreteProbDist


class CausalVariable(object):

    def __init__(self, name, prob_dist):
        """
        Class for summarizing functionalities of a single, causal variable. Each variable is
        described by a name and a conditional probability distribution.

        Parameters
        ----------
        name : str
               Name of the variable used for visualizing and inside other probability distributions.
        prob_dist : ProbDist
                    Object representing the conditional probability distribution of the variable.
        """
        super().__init__()
        self.name = name
        self.prob_dist = prob_dist

    def sample(self, inputs, *args, **kwargs):
        return self.prob_dist.sample(inputs, *args, **kwargs)

    def get_prob(self, inputs, output, *args, **kwargs):
        return self.prob_dist.prob(inputs, output, *args, **kwargs)

    def __str__(self):
        return "CausalVariable " + self.name

    def get_state_dict(self):
        state_dict = {"name": self.name}
        if self.prob_dist is not None:
            state_dict["prob_dist"] = self.prob_dist.get_state_dict()
            state_dict["prob_dist"]["class_name"] = str(self.prob_dist.__class__.__name__)
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        if "prob_dist" in state_dict:
            module = importlib.import_module("causal_graphs.variable_distributions")
            prob_dist_class = getattr(module, state_dict["prob_dist"]["class_name"])
            prob_dist = prob_dist_class.load_from_state_dict(state_dict["prob_dist"])
        else:
            prob_dist = None
        obj = CausalVariable(state_dict["name"], prob_dist)
        return obj


class CausalDAG(object):

    def __init__(self, variables, edges=None, adj_matrix=None, latents=None):
        """
        Main class for summarizing all functionalities and parameters of a causal graph. Each 
        causal graph consists of a set of variables and a graph structure description.

        Parameters
        ----------
        variables : list[CausalVariable]
                    A list of causal variables that are in the graph.
        edges : np.ndarray, shape [num_edges, 2], type np.int32
                A list of all edges in the graph. The graph structure needs to be described
                either as edge list or as adjacency matrix.
        adj_matrix : np.ndarray, shape [num_vars, num_vars], type np.bool
                     A matrix of ones and zeros where an entry (i,j) being one represents
                     an edge from variable i to variable j.
        latents : np.ndarray, shape [num_latents, 3]
                  A numpy array describing the latent confounders in the graph. If no latent
                  confounders are present, use None as input argument. Otherwise, the first
                  value in a row represents the variable index of the latent confounder, and
                  the consecutive two the indices of the two children.
        """
        super().__init__()
        assert len(set([v.name for v in variables])) == len(
            variables), "Variables need to have unique names to distinguish them."
        edges, adj_matrix = edges_or_adj_matrix(edges, adj_matrix, len(variables))

        self.variables = variables
        self.edges = edges
        self.adj_matrix = adj_matrix
        self.latents = latents if latents is not None else np.zeros((0, 3), dtype=np.int32)
        self.name_to_var = {v.name: v for v in variables}
        self._sort_variables()
        self.node_relations = get_node_relations(self.adj_matrix)

    def _sort_variables(self):
        """
        Sorts the variables in the graph for ancestral sampling.
        """
        self.variables, self.edges, self.adj_matrix, self.latents, _ = sort_graph_by_vars(
            self.variables, self.edges, self.adj_matrix, self.latents)

    def sample(self, interventions=None, batch_size=1, as_array=False):
        """
        Samples from the graph and conditional variable distributions according to ancestral sampling.

        Parameters
        ----------
        interventions : dict
                        Dictionary for specifing interventions that should be considered when sampling.
                        The keys should be variable names on which we intervene, and values can be
                        distributions in case of imperfect interventions, and values like a numpy array
                        otherwise. 
        batch_size : int
                     Number of samples to return.
        as_array : bool
                   If True, the samples are returned in one, stacked numpy array of 
                   shape [batch_size, num_vars]. Otherwise, the values are returned as dictionary of
                   variable_name -> samples.
        """

        if interventions is None:
            interventions = dict()

        var_vals = []
        for v_idx, var in enumerate(self.variables):
            parents = np.where(self.adj_matrix[:, v_idx])[0]
            parent_vals = {self.variables[i].name: var_vals[i] for i in parents}
            if interventions is None or (var.name not in interventions):  # No intervention
                sample = var.sample(parent_vals, batch_size=batch_size)
            elif isinstance(interventions[var.name], ProbDist):  # Imperfect intervention
                sample = interventions[var.name].sample(parent_vals, batch_size=batch_size)
            elif isinstance(var.prob_dist, DiscreteProbDist) and (interventions[var.name] == -1).any():  # -1 means resample
                sample = var.sample(parent_vals, batch_size=batch_size)
                sample = np.where(interventions[var.name] != -1, interventions[var.name], sample)
            else:  # Direct value assignment
                sample = interventions[var.name]
            var_vals.append(sample)

        if not as_array:
            var_vals = {var.name: var_vals[v_idx] for v_idx, var in enumerate(self.variables)}
        elif not isinstance(var_vals[0], np.ndarray):
            var_vals = np.array(var_vals)
        else:
            var_vals = np.stack(var_vals, axis=1)
        return var_vals

    def get_intervened_graph(self, interventions):
        """
        Returns the graph under the interventions given.

        Parameters
        ----------
        interventions : dict
                        Dictionary of variable_name -> intervention distribution/value. The distributions of
                        the variables in this dict will be replaced by the distribution in the dict, if 
                        interventions[variable_name] is a ProbDist object. Otherwise, it is assumed to be
                        a constant value and is assigned a ConstantDist object.
        """
        intervened_graph = deepcopy(self)
        for v_name in interventions:
            v_idx = [idx for idx, v in enumerate(intervened_graph.variables) if v.name == v_name][0]
            if isinstance(interventions[v_name], ProbDist):
                intervened_graph.variables[v_idx].prob_dist = interventions[v_name]
            else:
                intervened_graph.adj_matrix[:, v_idx] = False
                intervened_graph.variables[v_idx].prob_dist = ConstantDist(interventions[v_name])
        intervened_graph.edges = adj_matrix_to_edges(intervened_graph.adj_matrix)
        intervened_graph._sort_variables()
        return intervened_graph

    def __str__(self):
        """
        String description of the graph.
        """
        s = "CausalDAG with %i variables [%s]" % (len(self.variables), ",".join([v.name for v in self.variables]))
        s += " and %i edges%s\n" % (len(self.edges), ":" if len(self.edges) > 0 else "")
        for v_idx, v in enumerate(self.variables):
            children = np.where(self.adj_matrix[v_idx, :])[0]
            if len(children) > 0:
                s += "%s => %s" % (v.name, ",".join([self.variables[c].name for c in children])) + "\n"
        return s

    @property
    def num_vars(self):
        return len(self.variables)

    @property
    def num_latents(self):
        return self.latents.shape[0]

    def get_state_dict(self):
        """
        Returns a dictionary of all information that need to be stored to restore it at a later point.
        """
        state_dict = {"edges": self.edges,
                      "variables": [v.get_state_dict() for v in self.variables]}
        return state_dict

    def save_to_file(self, filename):
        """
        Saves the graph including all conditional distributions to disk.
        """
        torch.save(self.get_state_dict(), filename)

    @staticmethod
    def load_from_state_dict(state_dict):
        """
        Loads a graph object from a state dict exported by 'get_state_dict'.
        """
        edges = state_dict["edges"]
        variables = [CausalVariable.load_from_state_dict(v_dict) for v_dict in state_dict["variables"]]
        obj = CausalDAG(variables, edges)
        return obj

    @staticmethod
    def load_from_file(filename):
        """
        Loads a graph object from disk.
        """
        state_dict = torch.load(filename)
        return CausalDAG.load_from_state_dict(state_dict)


class CausalDAGDataset(CausalDAG):

    def __init__(self, adj_matrix, data_obs, data_int, latents=None):
        """
        A CausalDAG but with existing pre-sampled data and unknown conditional distributions.
        """
        num_categs = data_obs.max(axis=-1)
        variables = [CausalVariable(r"$X_{%i}$" % (i+1), CategoricalDist(num_categs[i]+1, None))
                     for i in range(adj_matrix.shape[0])]
        super().__init__(variables=variables, adj_matrix=adj_matrix, latents=latents)
        self.data_obs = data_obs  # Observational dataset, shape [num_samples, num_vars]
        self.data_int = data_int  # Interventional dataset, shape [num_vars, num_samples, num_vars]. First dim is the intervened variable.

    def sample(self, *args, **kwargs):
        raise Exception('You cannot generate new examples from a Causal-DAG dataset. '
                        'The specific distributions are unknown.')
