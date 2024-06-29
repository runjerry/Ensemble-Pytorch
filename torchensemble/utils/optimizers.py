import torch
from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}
        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

    # def __setstate__(self, state):
    #     super(SGD, self).__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class affineSGD(SGD):
    """Affine GTK SGD Optimizer. """

    def __init__(self, params, lr=required,
                 momentum=0, dampening=0, weight_decay=0,
                 use_bias=True, fullrank=True, scale=1.0,
                 fixed_rand_vec=True, weight_only=False,
                 same_norm=False, mat_norm=False, diag=False,
                 exact=False,
                 exponential=None, rank=2, diag_scale=1.0):

        assert not (fullrank and diag and (not exact)), (
            "fullrank and diag are incompatible with each other if not exact")

        super().__init__(params, lr=lr,
                         momentum=momentum,
                         dampening=dampening,
                         weight_decay=weight_decay)
        self._use_bias = use_bias
        self._exact = exact
        self._fullrank = fullrank
        self._scale = scale
        self._weight_only = weight_only
        self._same_norm = same_norm
        self._mat_norm = mat_norm
        self._diag = diag
        self._diag_scale = diag_scale
        self._exponential = exponential

        group = self.param_groups[0]
        params = group['params']
        self._depth = len(params)
        self._fixed_rand_vec = fixed_rand_vec
        if fixed_rand_vec:
            rand_vecs = []
            for idx, param in enumerate(params):
                print(f"========== Computing the {idx}-th rand_vec ============")
                if self._use_bias and idx % 2 == 1 and self._weight_only:
                    rand_vecs.append(None)
                else:
                    if self._exact:
                        size_param = param.data.nelement()
                        rand_mat = torch.rand(size_param, size_param, 
                                              device=param.data.device) 
                        mat_prod = rand_mat @ rand_mat.T + torch.eye(
                            size_param, device=rand_mat.device) * 1e-3
                        L = torch.linalg.cholesky(mat_prod)
                        mat_inv = torch.cholesky_inverse(L)
                        mat = mat_inv / mat_inv.norm()
                        rand_vecs.append(mat)
                    else:
                        last_dim = min(param.data.nelement(), rank)
                        # rand_vec = self._scale * torch.randn(
                        rand_vec = torch.randn(
                            [*param.data.shape, last_dim], device=param.data.device)
                        if self._diag:
                            if self._exponential:
                                rand_vec = torch.exp(self._exponential * rand_vec)
                            else:
                                rand_vec += 1
                                rand_vec = torch.clamp(rand_vec, 0.)
                        elif not self._same_norm and not self._mat_norm:
                            if rand_vec.ndim >= 4:
                                vec_norm = rand_vec.norm('nuc', dim=(2, 3), keepdim=True)
                            else:
                                vec_norm = rand_vec.norm()
                            rand_vec = rand_vec / vec_norm
                        elif self._mat_norm:
                            mat_rand_vec = rand_vec.reshape(-1, last_dim)
                            mat = mat_rand_vec @ mat_rand_vec.t()
                            vec_norm = torch.sqrt(mat.norm())
                            # elif rand_vec.ndim <= 2:
                            #     vec_norm = rand_vec.norm(dim=0, keepdim=True)
                            # else:
                            #     raise ValueError("rand_vec has wrong shape")
                            rand_vec = rand_vec / vec_norm
                        rand_vec *= self._scale
                        rand_vecs.append(rand_vec)
            self._rand_vecs = rand_vecs

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        params = group['params']
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']

        for idx, param in enumerate(params):
            if param is None:
                continue
            if self._use_bias and idx % 2 == 1 and self._weight_only:
                grad = param.grad.data
            else:
                if self._fixed_rand_vec:
                    rand_vec = self._rand_vecs[idx]
                else:
                    rand_vec = self._scale * torch.randn_like(
                        param.grad.data, device=param.data.device)
                    if self._diag:
                        if self._exponential:
                            rand_vec = torch.exp(self._exponential * rand_vec)
                        else:
                            rand_vec += 1
                            rand_vec = torch.clamp(rand_vec, 0.)
                    elif not self._same_norm and not self._mat_norm:
                        rand_vec = rand_vec / rand_vec.norm()
                if self._exact:
                    grad = param.grad.data.view(-1)
                    grad = rand_vec @ grad
                    grad = grad.reshape_as(param.grad.data)
                elif self._diag:
                    grad = rand_vec * param.grad.data
                else:
                    # grad = torch.mul(rand_vec, param.grad.data).sum() * rand_vec
                    prod = torch.einsum('...,...i->i', param.grad.data, rand_vec)
                    grad = torch.einsum('...i,i->...', rand_vec, prod)
                    if self._fullrank:
                        grad += self._diag_scale * param.grad.data
                # if self._same_norm:
                #     grad = (grad / grad.norm()) * param.grad.data.norm()

            if weight_decay != 0:
                grad = grad.add(param.data, alpha=weight_decay)
            if momentum != 0:
                param_state = self.state[param]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, grad)
                grad = buf

            param.data.add_(grad, alpha=-group['lr'])

        return loss
