"""
This code is mostly taken from https://github.com/slachapelle/dcdi/tree/master/dcdi.
It implements the Deep Sigmoidal Flow and Gaussian MLP for continuous data prediction.
"""
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm.auto import tqdm

import sys
sys.path.append('../')
from causal_discovery.multivariable_mlp import MultivarLinear, MultivarMLP, InputMask 


delta = 1e-6
c = - 0.5 * np.log(2 * np.pi)


def log(x):
    return torch.log(x * 1e2) - np.log(1e2)


def log_normal(x, mean, log_var, eps=0.00001):
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + c


def logsigmoid(x):
    return -softplus(-x)


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    def maximum(x):
        return x.max(axis)[0]

    A_max = oper(A, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - A_max), axis)

    B = torch.log(oper(A, summation, axis, True)) + A_max
    return B


def oper(array, oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for s in array.size():
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def softplus(x):
    return F.softplus(x) + delta


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


class BaseFlow(torch.nn.Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


class SigmoidFlow(BaseFlow):
    """
    Layer used to build Deep sigmoidal flows

    Parameters:
    -----------
    num_ds_dim: uint
        The number of hidden units

    """

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim

    def act_a(self, x):
        return softplus(x)

    def act_b(self, x):
        return x

    def act_w(self, x):
        return softmax(x, dim=2)

    def forward(self, x, logdet, dsparams, mollify=0.0, delta=delta):
        ndim = self.num_ds_dim
        # Apply activation functions to the parameters produced by the hypernetwork
        a_ = self.act_a(dsparams[:, :, 0: 1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim: 2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim: 3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b  # C
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)  # D
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)  # Logit function (so H)
        xnew = x_

        logj = F.log_softmax(dsparams[:, :, 2 * ndim: 3 * ndim], dim=2) + \
               logsigmoid(pre_sigm) + \
               logsigmoid(-pre_sigm) + log(a)

        logj = log_sum_exp(logj, 2).sum(2)

        logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
        logdet += logdet_

        return xnew, logdet


class DeepSigmoidalFlowModel(nn.Module):
    
    def __init__(self, num_vars, hidden_dims, flow_n_layers=1, flow_hid_dim=16):
        """
        Deep Sigmoidal Flow model

        :param int num_vars: number of variables
        :param int cond_n_layers: number of layers in the conditioner
        :param int cond_hid_dim: number of hidden units in the layers of the conditioner
        :param str cond_nonlin: type of non-linearity used in the conditioner
        :param int flow_n_layers: number of DSF layers
        :param int flow_hid_dim: number of hidden units in the DSF layers
        :param boolean intervention: True if use interventional version (DCDI)
        :param str intervention_type: Either perfect or imperfect
        :param str intervention_knowledge: Either known or unkown
        :param int num_regimes: total number of regimes in the data
        """
        super().__init__()
        flow_n_conditioned = flow_hid_dim

        # Conditioner model initialization
        n_conditioned_params = flow_n_conditioned * 3 * flow_n_layers  # Number of conditional params for each variable
        self.condition_mlp = MultivarMLP(input_dims=num_vars*2, 
                                         hidden_dims=hidden_dims, 
                                         output_dims=n_conditioned_params, 
                                         extra_dims=[num_vars],
                                         actfn=lambda : nn.LeakyReLU(inplace=True),
                                         pre_layers=InputMask(None, concat_mask=True))

        # Flow model initialization
        self.flow_n_layers = flow_n_layers
        self.flow_hid_dim = flow_hid_dim
        self.flow_n_params_per_var = flow_hid_dim * 3 * flow_n_layers  # total number of params
        self.flow_n_cond_params_per_var = n_conditioned_params  # number of conditional params
        self.flow_n_params_per_layer = flow_hid_dim * 3  # number of params in each flow layer
        self.flow = SigmoidFlow(flow_hid_dim)

        self.num_vars = num_vars

        # Shared density parameters (i.e, those that are not produced by the conditioner)
        self.shared_density_params = torch.nn.Parameter(torch.zeros(self.flow_n_params_per_var -
                                                                    self.flow_n_cond_params_per_var))
        self.reset_params()

    def reset_params(self):
        if "flow" in self.__dict__:
            self.flow.reset_parameters()
        if "shared_density_params" in self.__dict__:
            self.shared_density_params.data.uniform_(-0.001, 0.001)

    def forward(self, x, mask):
        """
        Compute the log likelihood of x given some density specification.

        :param x: torch.Tensor, shape=(batch_size, num_vars), the input for which to compute the likelihood.
        :param density_params: tuple of torch.Tensor, len=n_vars, shape of elements=(batch_size, n_flow_params_per_var)
            The parameters of the DSF model that were produced by the conditioner.
        :return: pseudo joint log-likelihood
        """
        # Convert the shape to (batch_size, n_vars, n_flow_params_per_var)
        density_params = self.condition_mlp(x, mask)

        assert len(density_params.shape) == 3
        assert density_params.shape[0] == x.shape[0]
        assert density_params.shape[1] == self.num_vars
        assert density_params.shape[2] == self.flow_n_cond_params_per_var

        # Inject shared parameters here
        # Add the shared density parameters in each layer's parameter vectors
        # The shared parameters are different for each layer
        # All batch elements receive the same shared parameters
        conditional = density_params.view(density_params.shape[0], density_params.shape[1], self.flow_n_layers, 3, -1)
        shared = \
            self.shared_density_params.view(self.flow_n_layers, 3, -1)[None, None, :, :, :].repeat(conditional.shape[0],
                                                                                                   conditional.shape[1],
                                                                                                   1, 1, 1)
        density_params = torch.cat((conditional, shared), -1).view(conditional.shape[0], conditional.shape[1], -1)
        assert density_params.shape[2] == self.flow_n_params_per_var

        logdet = Variable(torch.zeros((x.shape[0], self.num_vars), device=x.device))
        h = x.view(x.size(0), -1)
        for i in range(self.flow_n_layers):
            # Extract params of the current flow layer. Shape is (batch_size, n_vars, self.flow_n_params_per_layer)
            params = density_params[:, :, i * self.flow_n_params_per_layer: (i + 1) * self.flow_n_params_per_layer]
            h, logdet = self.flow(h, logdet, params)

        assert x.shape[0] == h.shape[0]
        assert x.shape[1] == h.shape[1]
        zeros = Variable(torch.zeros(x.shape[0], self.num_vars, device=x.device))
        # Not the joint NLL until we have a DAG
        pseudo_joint_nll = - log_normal(h, zeros, zeros + 1.0) - logdet
        return pseudo_joint_nll

    @property
    def device(self):
        return self.condition_mlp.device


class GaussianNoiseModel(nn.Module):

    def __init__(self, num_vars, hidden_dims):
        super().__init__()
        self.condition_mlp = MultivarMLP(input_dims=num_vars*2,  # Times two because of the mask
                                         hidden_dims=hidden_dims*2, 
                                         output_dims=2, 
                                         extra_dims=[num_vars],
                                         actfn=lambda : nn.LeakyReLU(inplace=True),
                                         pre_layers=InputMask(None, concat_mask=True))

    def forward(self, x, mask):
        y = self.condition_mlp(x, mask)
        nll = - log_normal(x, y[...,0], torch.exp(y[...,1]))
        return nll

    @property
    def device(self):
        return self.condition_mlp.device


def create_continuous_model(num_vars, hidden_dims, use_flow_model=False):
    if use_flow_model:
        flow = DeepSigmoidalFlowModel(num_vars, hidden_dims)
    else:
        flow = GaussianNoiseModel(num_vars, hidden_dims)
    return flow


if __name__ == '__main__':
    batch_size, num_vars = 32, 2
    torch.manual_seed(42)
    x = torch.randn(batch_size, num_vars)
    print(f'{x[0]=}')

    mask = (torch.arange(num_vars)[None] < torch.arange(num_vars)[:,None])[None].expand(batch_size, -1, -1)
    mask.fill_(1)
    print(f'{mask[0]=}')
    print('Creating model...')
    flow = create_flow_model(num_vars=num_vars, hidden_dims=[32,32])
    print('Running model...')
    nll = flow(x, mask)
    print(f'{nll[0]=}')
    x[0,0] = -10.0
    nll = flow(x, mask)
    print(f'{nll[0]=}')

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    t = tqdm(range(1000))
    for i in t:
        x = torch.randn(batch_size, num_vars)
        x = x.cumsum(dim=-1)

        nll = flow(x, mask).mean()
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
        t.set_description(f'Loss: {nll.item()}')
