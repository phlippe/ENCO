import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import sys
sys.path.append("../")

from causal_graphs.variable_distributions import _random_categ
from causal_discovery.datasets import InterventionalDataset


class GraphFitting(object):

    def __init__(self, model, graph, num_batches, num_graphs, theta_only_num_graphs, batch_size, lambda_sparse, max_graph_stacking=200):
        """
        Creates a DistributionFitting object that summarizes all functionalities
        for performing the graph fitting stage of ENCO.

        Parameters
        ----------
        model : MultivarMLP
                PyTorch module of the neural networks that model the conditional
                distributions.
        graph : CausalDAG
                Causal graph on which we want to perform causal structure learning.
        num_batches : int
                      Number of batches to use per MC sample in the graph fitting stage. 
                      Usually 1, only higher needed if GPU is running out of memory for 
                      common batch sizes.
        num_graphs : int 
                     Number of graph samples to use for estimating the gradients in the
                     graph fitting stage. Usually in the range 20-100.
        theta_only_num_graphs : int
                                Number of graph samples to use in the graph fitting stage if
                                gamma is frozen. Needs to be an even number, and usually 2 or 4.
        batch_size : int
                     Size of the batches to use in the gradient estimators.
        lambda_sparse : float
                        Sparsity regularizer value to use in the graph fitting stage.
        max_graph_stacking : int
                             Number of graphs that can maximally evaluated in parallel on the device.
                             If you run out of GPU memory, try to lower this number. It will then
                             evaluate the graph sequentially, which can be slightly slower but uses
                             less memory.
        """
        self.model = model
        self.graph = graph
        self.num_batches = num_batches
        self.num_graphs = num_graphs
        self.batch_size = batch_size
        self.lambda_sparse = lambda_sparse
        self.max_graph_stacking = max_graph_stacking
        self.theta_only_num_graphs = theta_only_num_graphs
        self.inter_vars = []
        if self.graph.num_vars >= 100 or hasattr(self.graph, "data_int"):
            self.dataset = InterventionalDataset(self.graph,
                                                 dataset_size=4096,
                                                 batch_size=self.batch_size)

    def perform_update_step(self, gamma, theta, var_idx=-1, only_theta=False):
        """
        Performs a full update step of the graph fitting stage. We first sample a batch of graphs,
        evaluate them on a interventional data batch, and estimate the gradients for gamma and theta
        based on the log-likelihoods. 

        Parameters
        ----------
        gamma : nn.Parameter
                Parameter tensor representing the gamma parameters in ENCO.
        theta : nn.Parameter
                Parameter tensor representing the theta parameters in ENCO.
        var_idx : int
                  Variable on which should be intervened to obtain the update. If none is given, i.e., 
                  a negative value, the variable will be randomly selected.
        only_theta : bool
                     If True, gamma is frozen and the gradients are only estimated for theta. See 
                     Appendix D.2 in the paper for details on the gamma freezing stage.
        """
        # Obtain log-likelihood estimates for randomly sampled graph structures
        if not only_theta:
            MC_samp = self.get_MC_samples(gamma, theta, num_batches=self.num_batches, num_graphs=self.num_graphs,
                                          batch_size=self.batch_size, var_idx=var_idx, mirror_graphs=False)
        else:
            MC_samp = self.get_MC_samples(gamma, theta, num_batches=self.num_batches, num_graphs=self.theta_only_num_graphs,
                                          batch_size=self.batch_size, var_idx=var_idx, mirror_graphs=True)
        adj_matrices, log_likelihoods, var_idx = MC_samp

        # Determine gradients for gamma and theta
        gamma_grads, theta_grads, theta_mask = self.gradient_estimator(
            adj_matrices, log_likelihoods, gamma, theta, var_idx)
        gamma.grad = gamma_grads
        theta.grad = theta_grads

        return theta_mask, var_idx

    @torch.no_grad()
    def get_MC_samples(self, gamma, theta, num_batches, num_graphs, batch_size,
                       var_idx=-1, mirror_graphs=False):
        """
        Samples and evaluates a batch of graph structures on a batch of interventional data.

        Parameters
        ----------
        gamma : nn.Parameter
                Parameter tensor representing the gamma parameters in ENCO.
        theta : nn.Parameter
                Parameter tensor representing the theta parameters in ENCO.
        num_batches : int
                      Number of batches to use per MC sample.
        num_graphs : int 
                     Number of graph structures to sample.
        batch_size : int
                     Size of interventional data batches.     
        var_idx : int
                  Variable on which should be intervened to obtain the update. If none is given, i.e., 
                  a negative value, the variable will be randomly selected.
        mirror_graphs : bool
                        This variable should be true if only theta is optimized. In this case, the first
                        half of the graph structure samples is identical to the second half, except that
                        the values of the outgoing edges of the intervened variable are flipped. This
                        allows for more efficient, low-variance gradient estimators. See details in 
                        the paper.
        """
        if mirror_graphs:
            assert num_graphs % 2 == 0, "Number of graphs must be divisible by two for mirroring"
        device = self.get_device()

        # Sample data batch
        if hasattr(self, "dataset"):
            # Pre-sampled data
            var_idx = self.sample_next_var_idx()
            int_sample = torch.cat([self.dataset.get_batch(var_idx) for _ in range(num_batches)], dim=0).to(device)
        else:
            # If no dataset exists, data is newly sampled from the graph
            intervention_dict, var_idx = self.sample_intervention(self.graph,
                                                                  dataset_size=num_batches*batch_size,
                                                                  var_idx=var_idx)
            int_sample = self.graph.sample(interventions=intervention_dict,
                                           batch_size=num_batches*batch_size,
                                           as_array=True)
            int_sample = torch.from_numpy(int_sample).long().to(device)

        # Split number of graph samples acorss multiple iterations if not all can fit into memory
        num_graphs_list = [min(self.max_graph_stacking, num_graphs-i*self.max_graph_stacking)
                           for i in range(math.ceil(num_graphs * 1.0 / self.max_graph_stacking))]
        num_graphs_list = [(num_graphs_list[i], sum(num_graphs_list[:i])) for i in range(len(num_graphs_list))]
        # Tensors needed for sampling
        edge_prob = (torch.sigmoid(gamma) * torch.sigmoid(theta)).detach()
        edge_prob_batch = edge_prob[None].expand(num_graphs, -1, -1)

        # Inner function for sampling a batch of random adjacency matrices from current belief probabilities
        def sample_adj_matrix():
            sample_matrix = torch.bernoulli(edge_prob_batch)
            sample_matrix = sample_matrix * (1 - torch.eye(sample_matrix.shape[-1], device=sample_matrix.device)[None])
            if mirror_graphs:  # First and second half of tensors are identical, except the intervened variable
                sample_matrix[num_graphs//2:] = sample_matrix[:num_graphs//2]
                sample_matrix[num_graphs//2:, var_idx] = 1 - sample_matrix[num_graphs//2:, var_idx]
                sample_matrix[:, var_idx, var_idx] = 0.
            return sample_matrix

        # Evaluate log-likelihoods under sampled adjacency matrix and data
        adj_matrices = []
        log_likelihoods = []
        for n_idx in range(num_batches):
            batch = int_sample[n_idx*batch_size:(n_idx+1)*batch_size]
            if n_idx == 0:
                adj_matrix = sample_adj_matrix()
                adj_matrices.append(adj_matrix)

            for c_idx, (graph_count, start_idx) in enumerate(num_graphs_list):
                adj_matrix_expanded = adj_matrix[start_idx:start_idx+graph_count,
                                                 None].expand(-1, batch_size, -1, -1).flatten(0, 1)
                batch_exp = batch[None, :].expand(graph_count, -1, -1).flatten(0, 1)
                nll = self.evaluate_likelihoods(batch_exp, adj_matrix_expanded, var_idx)
                nll = nll.reshape(graph_count, batch_size, -1)

                if n_idx == 0:
                    log_likelihoods.append(nll.mean(dim=1))
                else:
                    log_likelihoods[c_idx] += nll.mean(dim=1)

        # Combine all data
        adj_matrices = torch.cat(adj_matrices, dim=0)
        log_likelihoods = torch.cat(log_likelihoods, dim=0) / num_batches

        return adj_matrices, log_likelihoods, var_idx

    @torch.no_grad()
    def gradient_estimator(self, adj_matrices, log_likelihoods, gamma, theta, var_idx):
        """
        Returns the estimated gradients for gamma and theta. It uses the low-variance gradient estimators
        proposed in Section 3.3 of the paper. 

        Parameters
        ----------
        adj_matrices : torch.FloatTensor, shape [batch_size, num_vars, num_vars]
                       The adjacency matrices on which the interventional data has been evaluated on.
        log_likelihoods : torch.FloatTensor, shape [batch_size, num_vars]
                          The average log-likelihood under the adjacency matrices for all variables
                          in the graph.
        gamma : nn.Parameter
                Parameter tensor representing the gamma parameters in ENCO.
        theta : nn.Parameter
                Parameter tensor representing the theta parameters in ENCO.
        var_idx : int
                  Variable on which the intervention was performed. 
        """
        batch_size = adj_matrices.shape[0]
        log_likelihoods = log_likelihoods.unsqueeze(dim=1)

        orient_probs = torch.sigmoid(theta)
        edge_probs = torch.sigmoid(gamma)

        # Gradient calculation
        num_pos = adj_matrices.sum(dim=0)
        num_neg = batch_size - num_pos
        mask = ((num_pos > 0) * (num_neg > 0)).float()
        pos_grads = (log_likelihoods * adj_matrices).sum(dim=0) / num_pos.clamp_(min=1e-5)
        neg_grads = (log_likelihoods * (1 - adj_matrices)).sum(dim=0) / num_neg.clamp_(min=1e-5)
        gamma_grads = mask * edge_probs * (1 - edge_probs) * orient_probs * (pos_grads - neg_grads + self.lambda_sparse)
        theta_grads = mask * orient_probs * (1 - orient_probs) * edge_probs * (pos_grads - neg_grads)

        # Masking gamma for incoming edges to intervened variable
        gamma_grads[:, var_idx] = 0.
        gamma_grads[torch.arange(gamma_grads.shape[0]), torch.arange(gamma_grads.shape[1])] = 0.

        # Masking all theta's except the ones with a intervened variable
        theta_grads[:var_idx] = 0.
        theta_grads[var_idx+1:] = 0.
        theta_grads -= theta_grads.transpose(0, 1)  # theta_ij = -theta_ji

        # Creating a mask which theta's are actually updated for the optimizer
        theta_mask = torch.zeros_like(theta_grads)
        theta_mask[var_idx] = 1.
        theta_mask[:, var_idx] = 1.
        theta_mask[var_idx, var_idx] = 0.

        return gamma_grads, theta_grads, theta_mask

    def sample_next_var_idx(self):
        """
        Returns next variable to intervene on. We iterate through the variables
        in a shuffled order, like a standard dataset.
        """
        if len(self.inter_vars) == 0:  # If an epoch finished, reshuffle variables
            self.inter_vars = [i for i in range(len(self.graph.variables))]
            random.shuffle(self.inter_vars)
        var_idx = self.inter_vars.pop()
        return var_idx

    def sample_intervention(self, graph, dataset_size, var_idx=-1):
        """
        Returns a new data batch for an intervened variable.
        """
        # Select variable to intervene on
        if var_idx < 0:
            var_idx = self.sample_next_var_idx()
        var = graph.variables[var_idx]
        # Soft, perfect intervention => replace p(X_n) by random categorical
        # Scale is set to 0.0, which represents a uniform distribution.
        int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)
        # Sample from interventional distribution
        value = np.random.multinomial(n=1, pvals=int_dist, size=(dataset_size,))
        value = np.argmax(value, axis=-1)  # One-hot to index
        intervention_dict = {var.name: value}

        return intervention_dict, var_idx

    @torch.no_grad()
    def evaluate_likelihoods(self, int_sample, adj_matrix, var_idx):
        """
        Evaluates the negative log-likelihood of the interventional data batch (int_sample)
        on the given graph structures (adj_matrix) and the intervened variable (var_idx).
        """
        self.model.eval()
        device = self.get_device()
        int_sample = int_sample.to(device)
        adj_matrix = adj_matrix.to(device)
        # Transpose for mask because adj[i,j] means that i->j
        mask_adj_matrix = adj_matrix.transpose(1, 2)
        preds = self.model(int_sample, mask=mask_adj_matrix)

        # Evaluate negative log-likelihood of predictions
        preds = preds.flatten(0, 1)
        labels = int_sample.clone()
        labels[:, var_idx] = -1  # Perfect interventions => no predictions of the intervened variable
        labels = labels.reshape(-1)
        nll = F.cross_entropy(preds, labels, reduction='none', ignore_index=-1)
        nll = nll.reshape(*int_sample.shape)
        self.model.train()

        return nll

    def get_device(self):
        return self.model.device
