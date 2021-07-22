"""
This file contains the code for generating ground-truth conditional distributions.
Most experiments in the paper use the "NNCateg" distribution which is a randomly
initialized neural network.
"""
import numpy as np
import torch
import torch.nn as nn
from copy import copy
import sys
sys.path.append("../")
from causal_discovery.utils import get_device


class ProbDist(object):

    def __init__(self):
        """
        Abstract class representing a probability distribution. We want to sample from it, and
        eventually get the probability for an output.
        """
        pass

    def sample(self, inputs, batch_size=1):
        raise NotImplementedError

    def prob(self, inputs, output):
        raise NotImplementedError


####################
## DISCRETE PROBS ##
####################

class DiscreteProbDist(ProbDist):

    def __init__(self, val_range):
        """
        Abstract class of a discrete distribution (finite integer set or categorical).
        """
        super().__init__()
        self.val_range = val_range


class ConstantDist(DiscreteProbDist):

    def __init__(self, constant, val_range=None, **kwargs):
        """
        Represents a distribution that has a probability of 1.0 for one value, and zero for all others.
        """
        super().__init__(val_range=val_range)
        self.constant = constant

    def sample(self, inputs, batch_size=1):
        return np.repeat(self.constant, batch_size)

    def prob(self, inputs, output):
        if isinstance(output, np.ndarray):
            return (output == self.constant).astype(np.float32)
        else:
            return 1 if output == self.constant else 0

    def get_state_dict(self):
        # Export distribution
        state_dict = vars(self)
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = ConstantDist(state_dict["constant"], state_dict["val_range"])
        return obj


class CategoricalDist(DiscreteProbDist):

    def __init__(self, num_categs, prob_func, **kwargs):
        """
        Class representing a categorical distribution.

        Parameters
        ----------
        num_categs : int
                     Number of categories over which this distribution goes.
        prob_func : object (LeafCategDist, CategProduct, IndependentCategProduct, or NNCateg)
                    Object that describes the mapping of input categories to output probabilities.
        """
        super().__init__(val_range=(0, num_categs))
        self.num_categs = num_categs
        self.prob_func = prob_func

    def sample(self, inputs, batch_size=1):
        p = self.prob_func(inputs, batch_size)
        if len(p.shape) == 1:
            p = np.repeat(p[None], batch_size, axis=0)
        v = multinomial_batch(p)
        return v

    def prob(self, inputs, output):
        p = self.prob_func(inputs, batch_size=1)
        while len(p.shape) > 2:
            p = p[0]
        if len(p.shape) == 2:
            return p[np.arange(output.shape[0]), output]
        else:
            return p[..., output]

    def get_state_dict(self):
        # Export distribution including prob_func details.
        state_dict = {"num_categs": self.num_categs}
        if self.prob_func is not None:
            state_dict["prob_func"] = self.prob_func.get_state_dict()
            state_dict["prob_func"]["class_name"] = str(self.prob_func.__class__.__name__)
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        if "prob_func" in state_dict:
            prob_func_class = None
            if state_dict["prob_func"]["class_name"] == "LeafCategDist":
                prob_func_class = LeafCategDist
            elif state_dict["prob_func"]["class_name"] == "CategProduct":
                prob_func_class = CategProduct
            elif state_dict["prob_func"]["class_name"] == "IndependentCategProduct":
                prob_func_class = IndependentCategProduct
            elif state_dict["prob_func"]["class_name"] == "NNCateg":
                prob_func_class = NNCateg
            prob_func = prob_func_class.load_from_state_dict(state_dict["prob_func"])
        else:
            prob_func = None
        obj = CategoricalDist(state_dict["num_categs"], prob_func)
        return obj


class LeafCategDist(object):

    def __init__(self, num_categs, scale=1.0):
        """
        Random categorical distribution to represent prior distribution of leaf nodes.
        """
        self.probs = _random_categ(size=(num_categs,), scale=scale)
        self.num_categs = num_categs

    def __call__(self, inputs, batch_size):
        return self.probs

    def get_state_dict(self):
        state_dict = copy(vars(self))
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = LeafCategDist(state_dict["num_categs"])
        obj.probs = state_dict["probs"]
        return obj


class CategProduct(object):

    def __init__(self, input_names, input_num_categs=None, num_categs=None, val_grid=None, deterministic=False):
        """
        Categorical distribution with a random, independent distribution for every value pair of its parents.

        Parameters
        ----------
        input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
        input_num_categs : list[int]
                           Number of categories each input variable can take.
        num_categs : int
                     Number of categories over which the conditional distribution should be. 
        val_grid : np.ndarray, shape [input_num_categs[0], input_num_categs[1], ..., input_num_categs[-1], num_categs]
                   Array representing the probability distributions for each value pair of its parents. If 
                   None, a new val_grid is generated in this function.
        deterministic : bool
                        If True, we take the argmax over the generated val_grid, and assign a probability of
                        1.0 to the maximum value, all others zero.
        """
        if val_grid is None:
            assert input_num_categs is not None and num_categs is not None
            val_grid = _random_categ(size=tuple(input_num_categs) + (num_categs,))
            if deterministic:
                val_grid = (val_grid.max(axis=-1, keepdims=True) == val_grid).astype(np.float32)
        else:
            num_categs = val_grid.shape[-1]
            input_num_categs = val_grid.shape[:-1]
        self.val_grid = val_grid
        self.input_names = input_names
        self.input_num_categs = input_num_categs
        self.num_categs = num_categs

    def __call__(self, inputs, batch_size):
        idx = tuple([inputs[name] for name in self.input_names])
        v = self.val_grid[idx]
        return v

    def get_state_dict(self):
        state_dict = copy(vars(self))
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = CategProduct(state_dict["input_names"],
                           state_dict["input_num_categs"],
                           state_dict["num_categs"])
        obj.val_grid = state_dict["val_grid"]
        return obj


class IndependentCategProduct(object):

    def __init__(self, input_names, input_num_categs, num_categs,
                 scale_prior=0.2, scale_val=1.0):
        """
        Represents the conditional distribution as a product of independent conditionals per parent.
        For instance, the distribution p(A|B,C) is represented as:
                    p(A|B,C)=p_A(A)*p_B(A|B)*p_C(A|C)/sum_A[p_A(A)*p_B(A|B)*p_C(A|C)]
        
        Parameters
        ----------
        input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
        input_num_categs : list[int]
                           Number of categories each input variable can take.
        num_categs : int
                     Number of categories over which the conditional distribution should be. 
        scale_prior : float
                      Scale of the _random_categ distribution to use for the prior p_A(A)
        scale_val : float
                    Scale of the _random_categ distribution to use for all conditionals.
        """
        num_inputs = len(input_names)
        val_grids = [_random_categ(size=(c, num_categs), scale=scale_val) for c in input_num_categs]
        prior = _random_categ((num_inputs,), scale=scale_prior)
        self.val_grids = val_grids
        self.prior = prior
        self.num_categs = num_categs
        self.input_names = input_names
        self.input_num_categs = input_num_categs

    def __call__(self, inputs, batch_size):
        probs = np.zeros((batch_size, self.num_categs))
        for idx, name in enumerate(self.input_names):
            probs += self.prior[idx] * self.val_grids[idx][inputs[name]]
        return probs

    def get_state_dict(self):
        state_dict = copy(vars(self))
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = IndependentCategProduct(state_dict["input_names"],
                                      state_dict["input_num_categs"],
                                      state_dict["num_categs"])
        obj.prior = state_dict["prior"]
        obj.val_grids = state_dict["val_grids"]
        return obj


class NNCateg(object):

    def __init__(self, input_names, input_num_categs, num_categs):
        """
        Randomly initialized neural network that models an arbitrary conditional distribution.
        The network consists of a 2-layer network with LeakyReLU activation and an embedding
        layer for representing the categorical inputs. Weights are initialized with the 
        orthogonal initialization, and the biases uniform between -0.5 and 0.5.
        Architecture and initialization widely taken from Ke et al. (2020).

        Parameters
        ----------
        input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
        input_num_categs : list[int]
                           Number of categories each input variable can take.
        num_categs : int
                     Number of categories over which the conditional distribution should be. 
        """
        num_hidden = 48
        embed_dim = 4
        self.embed_module = nn.Embedding(sum(input_num_categs), embed_dim)
        self.net = nn.Sequential(nn.Linear(embed_dim*len(input_num_categs), num_hidden),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(num_hidden, num_categs, bias=False),
                                 nn.Softmax(dim=-1))
        for name, p in self.net.named_parameters():
            if name.endswith(".bias"):
                p.data.uniform_(-0.5, 0.5)
            else:
                nn.init.orthogonal_(p, gain=2.5)

        self.num_categs = num_categs
        self.input_names = input_names
        self.input_num_categs = input_num_categs
        self.device = get_device()
        self.embed_module.to(self.device)
        self.net.to(self.device)

    @torch.no_grad()
    def __call__(self, inputs, batch_size):
        inp_tensor = None
        for i, n, categs in zip(range(len(self.input_names)), self.input_names, self.input_num_categs):
            v = torch.from_numpy(inputs[n]).long()+sum(self.input_num_categs[:i])
            v = v.unsqueeze(dim=-1)
            inp_tensor = v if inp_tensor is None else torch.cat([inp_tensor, v], dim=-1)
        inp_tensor = inp_tensor.to(self.device)
        inp_tensor = self.embed_module(inp_tensor).flatten(-2, -1)
        probs = self.net(inp_tensor).cpu().numpy()
        return probs

    def get_state_dict(self):
        state_dict = copy(vars(self))
        state_dict["embed_module"] = self.embed_module.state_dict()
        state_dict["net"] = self.net.state_dict()
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = NNCateg(state_dict["input_names"],
                      state_dict["input_num_categs"],
                      state_dict["num_categs"])
        obj.embed_module.load_state_dict(state_dict["embed_module"])
        obj.net.load_state_dict(state_dict["net"])
        return obj


def multinomial_batch(p):
    # Effient batch-scale sampling in numpy
    u = np.random.uniform(size=p.shape[:-1]+(1,))
    p_cumsum = np.cumsum(p, axis=-1)
    diff = (p_cumsum - u)
    diff[diff < 0] = 2  # Set negatives to any number larger than 1
    samples = np.argmin(diff, axis=-1)
    return samples


#####################
## DIST GENERATORS ##
#####################

def _random_categ(size, scale=1.0, axis=-1):
    """
    Returns a random categorical distribution by sampling a value from a Gaussian distribution for each category, 
    and applying a softmax on those.

    Parameters
    ----------
    size - int / tuple
           For integer: Number of categories over which the distribution should be.
           For tuple: array size of samples from the Gaussian distribution
    scale - float
            Standard deviation to use for the Gaussian to sample from. scale=0.0 corresponds to a uniform 
            distribution. The larger the scale, the more peaked the distribution will be.
    axis - int
           If size is a tuple, axis specifies which axis represents the categories. The softmax is applied
           over the axis dimension.
    """
    val_grid = np.random.normal(scale=scale, size=size)
    val_grid = np.exp(val_grid)
    val_grid = val_grid / val_grid.sum(axis=axis, keepdims=True)
    return val_grid


def get_random_categorical(input_names, input_num_categs, num_categs, inputs_independent=True, use_nn=False, deterministic=False, **kwargs):
    """
    Returns a randomly generated, conditional distribution for categorical variables.

    Parameters
    ----------
    input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
                  Use an empty list to denote a leaf node distribution.
    input_num_categs : list[int]
                       Number of categories each input variable can take.
    num_categs : int
                 Number of categories over which the conditional distribution should be. 
    inputs_independent : bool
                         If True and not use_nn and not deterministic, the distribution is an IndependentCategProduct,
                         which models the distribution as product of independent conditionals.
    use_nn : bool
             If True and not deterministic, a randomly initialized neural network is used for generating the distribution.
    deterministic : bool
                    If True, the returned deterministic distribution will be deterministic. This means for every value
                    pair of the conditionals, there exists one category which has a probability 1.0, and all other
                    categories have a zero probability.
    """
    num_inputs = len(input_names)

    if num_inputs == 0:
        prob_func = LeafCategDist(num_categs)
    elif deterministic:
        prob_func = CategProduct(input_names, input_num_categs, num_categs, deterministic=deterministic)
    elif use_nn:
        prob_func = NNCateg(input_names, input_num_categs, num_categs)
    elif inputs_independent:
        prob_func = IndependentCategProduct(input_names, input_num_categs, num_categs)
    else:
        prob_func = CategProduct(input_names, input_num_categs, num_categs)

    return CategoricalDist(num_categs, prob_func, **kwargs)
