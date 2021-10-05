import torch
import torch.nn as nn
import math
import numpy as np


class MultivarMLP(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims, extra_dims, actfn, pre_layers=None):
        """
        Module for stacking N neural networks in parallel for more efficient evaluation. In the context
        of ENCO, we stack the neural networks of the conditional distributions for all N variables on top
        of each other to parallelize it on a GPU. 

        Parameters
        ----------
        input_dims : int
                     Input dimensionality for all networks (in ENCO, size of embedding)
        hidden_dims : list[int]
                      Hidden dimensionalities to use in the hidden layer. Length of list determines
                      the number of hidden layers to use.
        output_dims : int
                      Output dimensionality of all networks (in ENCO, max. number of categories)
        extra_dims : list[int]
                     Number of neural networks to have in parallel (in ENCO, number of variables).
                     Can have multiple dimensions if needed.
        actfn : function -> nn.Module
                Activation function to use in between hidden layers
        pre_layers : list[nn.Module]  / nn.Module
                     Any modules that should be applied before the actual MLP. This can include 
                     an embedding layer and/or masking operation.
        """
        super().__init__()
        self.extra_dims = extra_dims

        layers = []
        if pre_layers is not None:
            if not isinstance(pre_layers, list):
                layers += [pre_layers]
            else:
                layers += pre_layers
        hidden_dims = [input_dims] + hidden_dims
        for i in range(len(hidden_dims)-1):
            if not isinstance(layers[-1], EmbedLayer):  # After an embedding layer, we directly apply a non-linearity
                layers += [MultivarLinear(input_dims=hidden_dims[i],
                                          output_dims=hidden_dims[i+1],
                                          extra_dims=extra_dims)]
            layers += [actfn()]
        layers += [MultivarLinear(input_dims=hidden_dims[-1],
                                  output_dims=output_dims,
                                  extra_dims=extra_dims)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask=None):
        for l in self.layers:
            if isinstance(l, (EmbedLayer, InputMask)):
                x = l(x, mask=mask)
            else:
                x = l(x)
        return x

    @property
    def device(self):
        return next(iter(self.parameters())).device


class MultivarLinear(nn.Module):

    def __init__(self, input_dims, output_dims, extra_dims):
        """
        Linear layer with the same properties as MultivarMLP. It effectively applies N independent
        linear layers in parallel.

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        output_dims : int
                      Number of output dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.extra_dims = extra_dims

        self.weight = nn.Parameter(torch.zeros(*extra_dims, output_dims, input_dims))
        self.bias = nn.Parameter(torch.zeros(*extra_dims, output_dims))

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        # Shape preparation
        x_extra_dims = x.shape[1:-1]
        if len(x_extra_dims) > 0:
            for i in range(len(x_extra_dims)):
                assert x_extra_dims[-(i+1)] == self.extra_dims[-(i+1)], \
                    "Shape mismatch: X=%s, Layer=%s" % (str(x.shape), str(self.extra_dims))
        for _ in range(len(self.extra_dims)-len(x_extra_dims)):
            x = x.unsqueeze(dim=1)

        # Unsqueeze
        x = x.unsqueeze(dim=-1)
        weight = self.weight.unsqueeze(dim=0)
        bias = self.bias.unsqueeze(dim=0)

        # Linear layer
        out = torch.matmul(weight, x).squeeze(dim=-1)
        out = out + bias
        return out

    def extra_repr(self):
        # For printing
        return 'input_dims={}, output_dims={}, extra_dims={}'.format(
            self.input_dims, self.output_dims, str(self.extra_dims)
        )


class InputMask(nn.Module):

    def __init__(self, input_mask, concat_mask=False):
        """
        Module for handling to mask the input. Needed to simulate different parent sets.

        Parameters
        ----------
        input_mask : torch.Tensor/None
                     If a tensor, it is assumed to be a fixed mask for all forward passes.
                     If None, it is required to pass the mask during every forward pass.
        concat_mask : bool
                      If True, the mask will additionally be concatenated with the input.
                      Recommended for inputs where zero is a valid value
        """
        super().__init__()
        if isinstance(input_mask, torch.Tensor):
            self.register_buffer('input_mask', input_mask.float(), persistent=False)
        else:
            self.input_mask = input_mask
        self.concat_mask = concat_mask

    def forward(self, x, mask=None, mask_val=0):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input that should be masked.
        mask : torch.FloatTensor/None
               If self.input_mask is None, this tensor must be not none. Will be used
               to mask the input. A value of 1.0 means that an element is not masked,
               and 0.0 that it will be masked. Is broadcasted over dimensions with x.
        mask_val : float
                   Value to set for masked elements.
        """
        # Check if mask is passed or should be taken constant
        if mask is None:
            assert self.input_mask is not None, "No mask was given in InputMask module."
            mask = self.input_mask

        if len(mask.shape) > len(x.shape):
            x = x.reshape(x.shape[:1] + (1,)*(len(mask.shape)-len(x.shape)) + x.shape[1:])
        if len(x.shape) > len(mask.shape):
            mask = mask.reshape((1,)*(len(x.shape)-len(mask.shape)) + mask.shape)
        mask = mask.to(x.dtype)

        if mask_val != 0.0:
            x = x * mask + (1 - mask) * mask_val
        else:
            x = x * mask
        if self.concat_mask:
            x = torch.cat([x, mask.expand_as(x)], dim=-1)
        return x


class EmbedLayer(nn.Module):

    def __init__(self, num_vars, num_categs, hidden_dim, input_mask, sparse_embeds=False):
        """
        Embedding layer to represent categorical inputs in continuous space. For efficiency, the embeddings
        of different inputs are summed in this layer instead of stacked. This is equivalent to stacking the
        embeddings and applying a linear layer, but is more efficient with slightly more parameter cost.
        Masked inputs are represented by a zero embedding tensor.

        Parameters
        ----------
        num_vars : int
                   Number of variables that are input to each neural network.
        num_categs : int
                     Max. number of categories that each variable can take.
        hidden_dim : int
                     Output dimensionality of the embedding layer.
        input_mask : InputMask
                     Input mask module to use for masking possible inputs.
        sparse_embeds : bool
                        If True, we sparsify the embedding tensors before summing them together in the
                        forward pass. This is more memory efficient and can give a considerable speedup
                        for networks with many variables, but can be slightly slower for small networks.
                        It is recommended to set it to True for graphs with more than 50 variables.
        """
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.input_mask = input_mask
        self.sparse_embeds = sparse_embeds
        self.num_categs = num_categs
        # For each of the N networks, we have num_vars*num_categs possible embeddings to model.
        # Sharing embeddings across all N networks can limit the expressiveness of the networks.
        # Instead, we share them across 10-20 variables for large graphs to reduce memory.
        self.num_embeds = self.num_vars*self.num_vars*self.num_categs
        if self.num_embeds > 1e7:
            self.num_embeds = int(math.ceil(self.num_embeds / 20.0))
            self.shortend = True
        elif self.num_embeds > 1e6:
            for s in range(11, -1, -1):
                if self.num_vars % s == 0:
                    self.num_embeds = self.num_embeds // s
                    break
            self.shortend = True
        else:
            self.shortend = False
        self.embedding = nn.Embedding(num_embeddings=self.num_embeds,
                                      embedding_dim=hidden_dim)
        self.embedding.weight.data.mul_(2./math.sqrt(self.num_vars))
        self.bias = nn.Parameter(torch.zeros(num_vars, self.hidden_dim))
        # Tensor for mapping each input to its corresponding embedding range in self.embedding
        pos_trans = torch.arange(self.num_vars**2, dtype=torch.long) * self.num_categs
        self.register_buffer("pos_trans", pos_trans, persistent=False)

    def forward(self, x, mask):
        # For very large x tensors during graph fitting, it is more efficient to split it
        # into multiple sub-tensors before running the forward pass.
        num_chunks = int(math.ceil(np.prod(mask.shape) / 256e5))
        if self.training or num_chunks == 1:
            return self.embed_tensor(x, mask)
        else:
            x = x.chunk(num_chunks, dim=0)
            mask = mask.chunk(num_chunks, dim=0)
            x_out = []
            for x_l, mask_l in zip(x, mask):
                out_l = self.embed_tensor(x_l, mask_l)
                x_out.append(out_l)
            x_out = torch.cat(x_out, dim=0)
            return x_out

    def embed_tensor(self, x, mask):
        assert x.shape[-1] == self.num_vars
        if len(x.shape) == 2:  # Add variable dimension
            x = x.unsqueeze(dim=1).expand(-1, self.num_vars, -1)
        else:
            assert x.shape[-2] == self.num_vars

        # Number of variables
        pos_trans = self.pos_trans.view((1,)*(len(x.shape)-2) + (self.num_vars, self.num_vars))
        x = x + pos_trans

        if self.sparse_embeds:
            # Selects the non-zero embedding tensors and stores them in a separate tensor instead of masking.
            # Lower memory consumption and faster for networks with many variables.
            flattened_mask = mask.flatten(0, 1).long()
            num_neighbours = flattened_mask.sum(dim=-1)
            max_neighbours = num_neighbours.max()
            x_sparse = torch.masked_select(x, mask == 1.0)
            if self.shortend:
                x_sparse = x_sparse % self.num_embeds
            x_sparse = self.embedding(x_sparse)
            x_sparse = torch.cat([x_sparse.new_zeros(x_sparse.shape[:-2]+(1,)+x_sparse.shape[-1:]), x_sparse], dim=-2)
            idxs = flattened_mask.cumsum(dim=-1)
            idxs[1:] += num_neighbours[:-1].cumsum(dim=-1)[..., None]
            idxs = (idxs * flattened_mask).sort(dim=-1, descending=True)[0]
            # Determine how many embeddings to sum per variable. Needed to construct the sparse tensor.
            sort_neighbours, sort_indices = num_neighbours.sort(dim=0)
            _, resort_indices = sort_indices.sort(dim=0)
            pos = 1+torch.arange(num_neighbours.shape[0], device=num_neighbours.device, dtype=torch.long)
            comp_cost = sort_neighbours * pos + max_neighbours * (num_neighbours.shape[0] - pos)
            min_cost, argmin_cost = comp_cost.min(dim=0)
            mid_neighbours = sort_neighbours[argmin_cost]
            # More efficient: split tensor into two, one half with the variables with the least and the other
            # with the most embeddings to sum. This prevents large computational costs if we have a few outliers.
            idxs = idxs[sort_indices]
            idxs = idxs[:, :max_neighbours]
            if mid_neighbours > 0:
                x_new_1 = x_sparse.index_select(index=idxs[:argmin_cost+1, :mid_neighbours].reshape(-1), dim=0)
                x_1 = x_new_1.reshape(-1, mid_neighbours, x_sparse.shape[-1]).sum(dim=-2)
            else:
                x_1 = x_sparse.new_zeros(argmin_cost+1, x_sparse.shape[-1])
            x_new_2 = x_sparse.index_select(index=idxs[argmin_cost+1:, :max_neighbours].reshape(-1), dim=0)
            x_2 = x_new_2.reshape(-1, max_neighbours, x_sparse.shape[-1]).sum(dim=-2)
            # Bring tensors back in order
            x = torch.cat([x_1, x_2], dim=0)[resort_indices]
            x = x.reshape(mask.shape[0], mask.shape[1], x.shape[-1])
        else:
            if self.shortend:
                x = x % self.num_embeds
            x = self.embedding(x)
            x = self.input_mask(x, mask=mask[..., None], mask_val=0.0)

        if len(x.shape) > 3:
            x = x.sum(dim=-2)

        bias = self.bias.view((1,)*(len(x.shape)-2) + self.bias.shape)
        x = x + bias
        return x


def get_activation_function(actfn):
    """
    Returns an activation function based on a string description.
    """
    if actfn is None or actfn == 'leakyrelu':
        def create_actfn(): return nn.LeakyReLU(0.1, inplace=True)
    elif actfn == 'gelu':
        def create_actfn(): return nn.GELU()
    elif actfn == 'relu':
        def create_actfn(): return nn.ReLU()
    elif actfn == 'swish' or actfn == 'silu':
        def create_actfn(): return nn.SiLU()
    else:
        raise Exception('Unknown activation function ' + str(actfn))
    return create_actfn


def create_model(num_vars, num_categs, hidden_dims, actfn=None):
    """
    Method for creating a full multivariable MLP as used in ENCO.
    """
    num_outputs = max(1, num_categs)
    num_inputs = num_vars
    actfn = get_activation_function(actfn)

    mask = InputMask(None)
    if num_categs > 0:
        pre_layers = EmbedLayer(num_vars=num_vars,
                                num_categs=num_categs,
                                hidden_dim=hidden_dims[0],
                                input_mask=mask,
                                sparse_embeds=(num_vars >= 50))
        num_inputs = pre_layers.hidden_dim
        pre_layers = [pre_layers, actfn()]
    else:
        pre_layers = mask

    mlps = MultivarMLP(input_dims=num_inputs,
                       hidden_dims=hidden_dims,
                       output_dims=num_outputs,
                       extra_dims=[num_vars],
                       actfn=actfn,
                       pre_layers=pre_layers)
    return mlps
