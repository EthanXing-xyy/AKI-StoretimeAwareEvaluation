import torch
from torch.optim import Optimizer

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    SAM seeks parameters that lie in neighborhoods having uniformly low loss.
    This helps improve model generalization by finding flatter minima.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Initialize SAM optimizer.

        Args:
            params: Model parameters to optimize
            base_optimizer: Base optimizer (e.g., SGD, Adam)
            rho: Neighborhood size for perturbation
            adaptive: Whether to use adaptive SAM (ASAM)
            **kwargs: Additional arguments for base optimizer
        """
        if rho < 0.0:
            raise ValueError("Invalid rho, should be >= 0.0")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step of SAM: Compute gradient norm and perturb weights.
        This step moves parameters in the direction of steepest ascent.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (p ** 2 if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step of SAM: Restore weights and update using perturbed gradients.
        This step moves parameters in the direction of steepest descent.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"

        with torch.enable_grad():
            loss = closure()

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()

        self.second_step()
        return loss

    def _grad_norm(self):
        """
        Compute the L2 norm of gradients across all parameters.
        For adaptive SAM, weights are scaled by their absolute values.
        """
        device = self.param_groups[0]['params'][0].device
        shared_device = device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
