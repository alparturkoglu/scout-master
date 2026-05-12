import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, Normal, Exponential


from nflows import transforms, distributions, flows
from nflows.transforms.base import Transform, InputOutsideDomain, InverseTransform




class MyNFlows(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

        # Cache used to avoid per-forward device sync overhead.
        self._synced_device = None

        self.transform = transforms.CompositeTransform([
            # transforms.LogTransform(),  # log(x)
            transforms.ActNorm(features=d),
            transforms.Sigmoid(),
            transforms.PiecewiseRationalQuadraticCDF(shape=[d], num_bins=16),
            transforms.Logit(),
        ])
        self.base = distributions.StandardNormal(shape=[d])
        self.flow = flows.Flow(transform=self.transform, distribution=self.base)

    def _sync_nflows_tensors_to_device(self, device: torch.device) -> None:
        """nflows Sigmoid/Logit store `temperature` as a plain CPU tensor when
        `learn_temperature=False`, so `.to(device)` doesn't move it.

        To keep the fast original nflows transforms, we move those tiny tensors onto
        the input device once per device.
        """

        def _recurse(t):
            if hasattr(t, "temperature") and isinstance(t.temperature, torch.Tensor):
                if t.temperature.device != device:
                    t.temperature = t.temperature.to(device)

            # InverseTransform-like wrappers.
            inner = getattr(t, "_transform", None)
            if inner is not None:
                _recurse(inner)

            # CompositeTransform stores children under `_transforms`.
            children = getattr(t, "_transforms", None)
            if children is not None:
                for child in children:
                    _recurse(child)

        _recurse(self.flow._transform)

    def forward(self, e):
        if self._synced_device != e.device:
            self._sync_nflows_tensors_to_device(e.device)
            self._synced_device = e.device
        z, log_det_jacobian = self.flow._transform.forward(e)
        return z, log_det_jacobian
    
    def inverse(self, z):
        # Inverse mapping: latent z -> data x
        x, log_det_jacobian = self.flow._transform.inverse(z)
        return x, log_det_jacobian
