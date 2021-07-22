import torch


class OptimizerTemplate:

    def __init__(self, params, lr):
        """
        Optimizer template for a single parameter tensor.

        Parameters
        ----------
        params : nn.Parameter / torch.Tensor with grads
                 The parameters that should be optimized
        lr : float
             Basic learning rate for the optimizer.
        """
        self.params = params
        self.lr = lr

    def zero_grad(self):
        # Set gradients of all parameters to zero
        if self.params.grad is not None:
            self.params.grad.detach_()
            self.params.grad.zero_()


class AdamTheta(OptimizerTemplate):

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam optimizer for the theta parameters. The difference to standard Adam is that not all 
        parameters are optimized in every step, but many are masked out. Hence, only a part of the
        parameters are updated, including their first- and second-order momentum.

        Parameters
        ----------
        params : nn.Parameter / torch.Tensor with grads
                 The parameters that should be optimized, here representing theta of ENCO.
        lr : float
             Basic learning rate of Adam.
        beta1 : float
                beta-1 hyperparameter of Adam.
        beta2 : float
                beta-2 hyperparameter of Adam.
        eps : float
              Epsilon hyperparameter of Adam.
        """
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = torch.zeros_like(self.params.data)  # Remembers "t" for each parameter for bias correction
        self.param_momentum = torch.zeros_like(self.params.data)
        self.param_2nd_momentum = torch.zeros_like(self.params.data)

    @torch.no_grad()
    def step(self, mask):
        """
        Standard Adam update step, except that only a subset of the parameters is updated.
        The subset is determined by the given mask.

        Parameters
        ----------
        mask : torch.FloatTensor, shape equal to self.params
               A mask with values being 0 or 1. If the value at position (i,j) is 1, the
               parameter self.params[i,j] will be updated in this step. Otherwise, it is
               not changed.
        """
        if self.params.grad is None:
            return

        self.param_step.add_(mask)

        new_momentum = (1 - self.beta1) * self.params.grad + self.beta1 * self.param_momentum
        new_2nd_momentum = (1 - self.beta2) * (self.params.grad)**2 + self.beta2 * self.param_2nd_momentum
        self.param_momentum = torch.where(mask == 1.0, new_momentum, self.param_momentum)
        self.param_2nd_momentum = torch.where(mask == 1.0, new_2nd_momentum, self.param_2nd_momentum)

        bias_correction_1 = 1 - self.beta1 ** self.param_step
        bias_correction_2 = 1 - self.beta2 ** self.param_step
        bias_correction_1.masked_fill_(bias_correction_1 == 0.0, 1.0)
        bias_correction_2.masked_fill_(bias_correction_2 == 0.0, 1.0)

        p_2nd_mom = self.param_2nd_momentum / bias_correction_2
        p_mom = self.param_momentum / bias_correction_1
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom
        p_update = mask * p_update

        self.params.add_(p_update)

    @torch.no_grad()
    def to(self, device):
        self.param_step = self.param_step.to(device)
        self.param_momentum = self.param_momentum.to(device)
        self.param_2nd_momentum = self.param_2nd_momentum.to(device)


class AdamGamma(OptimizerTemplate):

    def __init__(self, params, lr, beta1=0.9, beta2=0.9, eps=1e-8):
        """
        Adam optimizer for the gamma parameters when latent confounders should be detected. 
        The difference to standard Adam is that we track the gradients and first-order 
        momentum parameter for observational and interventional data separately. After
        training, the latent confounder scores can be calculated from the aggregated 
        gradients. The difference of gamma optimized via this optimizer vs standard Adam
        is neglectable, and did not show any noticable differences in performance. 

        Parameters
        ----------
        params : nn.Parameter / torch.Tensor with grads
                 The parameters that should be optimized, here representing gamma of ENCO.
        lr : float
             Basic learning rate of Adam.
        beta1 : float
                beta-1 hyperparameter of Adam.
        beta2 : float
                beta-2 hyperparameter of Adam.
        eps : float
              Epsilon hyperparameter of Adam.
        """
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = torch.zeros(self.params.data.shape + (2,), device=self.params.device)
        self.param_momentum = torch.zeros(self.params.data.shape + (2,), device=self.params.device)
        self.param_2nd_momentum = torch.zeros_like(self.params.data)  # Adaptive lr needs to shared of obs and int data
        self.updates = torch.zeros(self.params.data.shape + (2,), device=self.params.device)

    @torch.no_grad()
    def step(self, var_idx):
        """
        Standard Adam update step, except that it tracks the gradients of the variable 'var_idx'
        separately from all other variables.

        Parameters
        ----------
        var_idx : int
                  Index of the variable on which an intervention has been performed. The input 
                  should be negative in case no intervention had been performed.
        """
        if self.params.grad is None:
            return

        mask = torch.ones_like(self.params.data)
        mask_obs_int = torch.ones_like(self.param_step)
        if var_idx >= 0:
            mask[:, var_idx] = 0.0
            mask_obs_int[var_idx, :, 0] = 0.0
            mask_obs_int[..., 1] -= mask_obs_int[..., 0]
            mask_obs_int[:, var_idx, :] = 0.0

        self.param_step.add_(mask_obs_int)

        new_momentum = (1 - self.beta1) * self.params.grad[..., None] + self.beta1 * self.param_momentum
        new_2nd_momentum = (1 - self.beta2) * (self.params.grad)**2 + self.beta2 * self.param_2nd_momentum
        self.param_momentum = torch.where(mask_obs_int == 1.0, new_momentum, self.param_momentum)
        self.param_2nd_momentum = torch.where(mask == 1.0, new_2nd_momentum, self.param_2nd_momentum)

        bias_correction_1 = 1 - self.beta1 ** self.param_step
        bias_correction_2 = 1 - self.beta2 ** self.param_step.sum(dim=-1)
        bias_correction_1.masked_fill_(bias_correction_1 == 0.0, 1.0)
        bias_correction_2.masked_fill_(bias_correction_2 == 0.0, 1.0)

        p_2nd_mom = self.param_2nd_momentum / bias_correction_2
        p_mom = self.param_momentum / bias_correction_1
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr[..., None] * p_mom
        p_update = mask_obs_int * p_update

        self.params.add_(p_update.sum(dim=-1))
        self.updates.add_(p_update)

    @torch.no_grad()
    def to(self, device):
        self.param_step = self.param_step.to(device)
        self.param_momentum = self.param_momentum.to(device)
        self.param_2nd_momentum = self.param_2nd_momentum.to(device)
        self.updates = self.updates.to(device)
