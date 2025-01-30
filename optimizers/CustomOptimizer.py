import torch
from torch.optim import Optimizer

class CustomOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        # Set default values for learning rate, momentum, and weight decay
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(CustomOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Access hyperparameters from the group
                lr = group['lr']
                momentum = group['momentum']
                weight_decay = group['weight_decay']

                # Get the gradient
                grad = p.grad.data

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Update momentum (if applicable)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    grad = buf

                # Update the parameter
                p.data.add_(grad, alpha=-lr)

        return loss