import torch
import torch.nn as nn


class DistributionFitting(object):

    def __init__(self, model, optimizer, data_loader):
        """
        Creates a DistributionFitting object that summarizes all functionalities
        for performing the distribution fitting stage of ENCO.

        Parameters
        ----------
        model : MultivarMLP
                PyTorch module of the neural networks that model the conditional
                distributions.
        optimizer : torch.optim.Optimizer
                    Standard PyTorch optimizer for the model.
        data_loader : torch.data.DataLoader
                      Data loader returning batches of observational data. This
                      data is used for training the neural networks.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_module = nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def _get_next_batch(self):
        """
        Helper function for sampling batches one by one from the data loader.
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch

    def perform_update_step(self, sample_matrix):
        """
        Performs a full update step of the distribution fitting stage.
        It first samples a batch of random adjacency matrices from 'sample_matrix',
        and then performs a training step on a random observational data batch.

        Parameters
        ----------
        sample_matrix : torch.FloatTensor, shape [num_vars, num_vars]
                        Float tensor with values between 0 and 1. An element (i,j)
                        represents the probability of having an edge from X_i to X_j,
                        i.e., not masking input X_i for predicting X_j.

        Returns
        -------
        loss : float
               The loss of the model with the sampled adjacency matrices on the
               observational data batch.
        """
        batch = self._get_next_batch()
        adj_matrices = self.sample_graphs(sample_matrix=sample_matrix,
                                          batch_size=batch.shape[0])
        loss = self.train_step(batch, adj_matrices)
        return loss

    @torch.no_grad()
    def sample_graphs(self, sample_matrix, batch_size):
        """
        Samples a batch of adjacency matrices that are used for masking the inputs.
        """
        sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
        adj_matrices = torch.bernoulli(sample_matrix)
        # Mask diagonals
        adj_matrices[:, torch.arange(adj_matrices.shape[1]), torch.arange(adj_matrices.shape[2])] = 0
        return adj_matrices

    def train_step(self, inputs, adj_matrices):
        """
        Performs single optimization step of the neural networks
        on given inputs and adjacency matrix.
        """
        self.model.train()
        self.optimizer.zero_grad()
        device = self.model.device
        inputs = inputs.to(device)
        adj_matrices = adj_matrices.to(device)
        # Transpose for mask because adj[i,j] means that i->j
        mask_adj_matrices = adj_matrices.transpose(1, 2)
        preds = self.model(inputs, mask=mask_adj_matrices)

        if inputs.dtype == torch.long:
            loss = self.loss_module(preds.flatten(0,-2), inputs.reshape(-1))
        else:  # If False, our input was continuous, and we return log likelihoods as preds
            loss = preds.mean()

        loss.backward()
        self.optimizer.step()

        return loss.item()
