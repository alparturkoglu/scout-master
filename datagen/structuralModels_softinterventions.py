from matplotlib.style import available
import networkx as nx
import numpy as np 
import torch
import math

from models.functions_softinterventions import indMLPFunction, nonlinearMLP, NoisyFunction

def standard_normal_logprob(z, noise_scale=0.5):
    logZ = -0.5 * math.log(2 * math.pi * noise_scale**2)
    return logZ - z**2 / (2 * noise_scale**2)

def make_non_cotractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale = 1.0
    if s[0] <= 1.0:
        scale = 2/s[0]
    
    return scale * weights 

def make_contractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale=1.1
    if s[0] >= 1.0:
        scale = 1.1 * s[0]
    
    return weights/scale


class linearSEM:

    """
    -------------------------------------------------------------------
    This class models a Linear Structural Equation Model (Linear SEM)
    -------------------------------------------------------------------
    The model is initialized with the number of nodes in the graph and
    the absolute minimum and maximum weights for the edges. 
    """
    def __init__(self, graph, abs_weight_low=0.2, abs_weight_high=0.9, noise_scale=0.5, contractive=True, noisy=False, noisy_weight=-1):
        self.graph = graph
        self.abs_weight_low = abs_weight_low 
        self.abs_weight_high = abs_weight_high
        self.contractive = contractive

        self.n_nodes = len(graph.nodes)
        
        self.weights = np.random.uniform(self.abs_weight_low, self.abs_weight_high, size=(self.n_nodes, self.n_nodes))
        self.weights *= 2 * np.random.binomial(1, 0.5, size=self.weights.shape) - 1
        adjacency = nx.to_numpy_array(self.graph)
        self.weights *= adjacency

        self.noise_scale = noise_scale

        self.noisy = noisy
        self.noisy_weight = noisy_weight


        if not self.contractive:
            self.weights = make_non_cotractive(self.weights)
        else:
            self.weights = make_contractive(self.weights)

        if self.noisy:
            self.weights_i = self.noisy_weight * self.weights
        else:
            self.weights_i = self.weights
       
        if not self.contractive:
            self.weights_i = make_non_cotractive(self.weights_i)
        else:
            self.weights_i = make_contractive(self.weights_i)

    def generateData(self, n_samples, intervention_set=[None], lat_provided=False, latent_vec=None, fixed_intervention=False, return_latents=False, shift_scale=0, intervention_scale=0.5, noise_type='gaussian'):
        # set intervention_set = [None] for purely observational data.
        self.shift_scale = shift_scale
        self.intervention_scale = intervention_scale
        self.noise_type = noise_type

        observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
        U = np.zeros((self.n_nodes, self.n_nodes))
        U[observed_set, observed_set] = 1

        I = np.eye(self.n_nodes)

        
        if lat_provided:
            E = latent_vec.T
        else:
            E = np.zeros((self.n_nodes, n_samples))
            if self.noise_type == 'gaussian':
                if len(observed_set) > 0:
                    E[observed_set,:] = self.noise_scale * np.random.randn(len(observed_set), n_samples)
                if intervention_set[0] != None:
                    E[intervention_set,:] = self.intervention_scale * np.random.randn(len(intervention_set), n_samples) + self.shift_scale
            elif self.noise_type == 'exponential':
                if len(observed_set) > 0:
                    E[observed_set,:] = np.random.exponential(scale=self.noise_scale, size=(len(observed_set), n_samples))
                if intervention_set[0] != None:
                    E[intervention_set,:] = np.random.exponential(scale=self.intervention_scale, size=(len(intervention_set), n_samples)) + self.shift_scale
            elif self.noise_type == 'gumbel':
                if len(observed_set) > 0:
                    E[observed_set,:] = np.random.gumbel(loc=0.0, scale=self.noise_scale, size=(len(observed_set), n_samples))
                if intervention_set[0] != None:
                    E[intervention_set,:] = np.random.gumbel(loc=0.0, scale=self.intervention_scale, size=(len(intervention_set), n_samples)) + self.shift_scale
            elif self.noise_type == 'laplace':
                if len(observed_set) > 0:
                    E[observed_set,:] = np.random.laplace(loc=0.0, scale=self.noise_scale, size=(len(observed_set), n_samples))
                if intervention_set[0] != None:
                    E[intervention_set,:] = np.random.laplace(loc=0.0, scale=self.intervention_scale, size=(len(intervention_set), n_samples)) + self.shift_scale
            else:
                raise ValueError(f"Unknown noise_type: {self.noise_type}")

        X = np.linalg.inv(I - (U @ self.weights.T) - ((I-U) @ self.weights_i.T)) @ (E)

        # The final data matrix is dimensions - n_samples X self.nodes
        if return_latents:
            return X.T, E.T
            
        return X.T

class nonlinearSEM:
    """
    ----------------------------------------------------------------------
    This class models a Nonlinear Structural Equation Model (Linear SEM)
    ----------------------------------------------------------------------
    The nonlinear function is taken from models.functions 
    """

    def __init__(self, graph, lip_const=0.9, fun_type='sin-mlp', act_fun='tanh', device=None, noise_scale=0.5, n_hidden=1, bias=False, contractive=True, noisy=False, noisy_weight=-1, variable_shift_scale=False, shift_scale_range=None, variable_intervention_scale=False, intervention_scale_range=None):
        self.lip_const = lip_const 
        self.graph = graph 
        self.n_nodes = len(graph.nodes)
        self.act_fun = act_fun
        self.n_hidden = n_hidden
        self.bias = bias
        self.noisy = noisy  # whether to use noisy function for interventions
        self.contractive = contractive 
        self.noisy_weight = noisy_weight
        self.variable_shift_scale = variable_shift_scale
        self.shift_scale_range = shift_scale_range
        self.variable_intervention_scale = variable_intervention_scale
        self.intervention_scale_range = intervention_scale_range
        self.last_intervention_shifts = None
        self.last_intervention_scales = None
        if self.contractive:
            self.lip_const = 2.0

        if fun_type == 'mul-mlp':
            self.f = indMLPFunction(n_nodes=self.n_nodes, 
                                    lip_constant=self.lip_const,
                                    activation=self.act_fun,
                                    n_layers=n_hidden,
                                    full_input=False,
                                    graph_given=True,
                                    graph=self.graph, 
                                    bias=self.bias)

        else:
            self.f = nonlinearMLP(n_nodes=self.n_nodes, 
                                  lip_constant=self.lip_const,
                                  n_layers=self.n_hidden, 
                                  bias=self.bias,
                                  activation_fn=self.act_fun, 
                                  graph_given=True, 
                                  graph=self.graph)

        if self.noisy == True:
            self.f_i = NoisyFunction(self.f, noise_scale=1, noise_mean=0, noisy_weight=self.noisy_weight)
        else:
            self.f_i = self.f  # initially set to be the same as f

        if device is not None:
            self.f = self.f.to(device)
            self.f_i = self.f_i.to(device)
        self.device = device
        self.noise_scale = noise_scale

        
    def generateData(self, n_samples, intervention_set=[None], lat_provided=False, latent_vec=None, n_iter=30, fixed_intervention=False, return_latents=False, intervention_scale=0.5, shift_scale=0, noise_type='gaussian', variable_shift_scale=None, shift_scale_range=None, variable_intervention_scale=None, intervention_scale_range=None):
        # set intervention_set = [None] for purely observational data
        self.intervention_scale = intervention_scale
        self.shift_scale = shift_scale
        self.noise_type = noise_type

        effective_variable_shift_scale = self.variable_shift_scale if variable_shift_scale is None else variable_shift_scale
        effective_shift_scale_range = self.shift_scale_range if shift_scale_range is None else shift_scale_range
        effective_variable_intervention_scale = self.variable_intervention_scale if variable_intervention_scale is None else variable_intervention_scale
        effective_intervention_scale_range = self.intervention_scale_range if intervention_scale_range is None else intervention_scale_range

        shift_term = float(self.shift_scale)
        if intervention_set[0] is not None and effective_variable_shift_scale:
            if effective_shift_scale_range is None:
                bound = abs(float(self.shift_scale))
                low, high = -bound, bound
            else:
                low, high = float(effective_shift_scale_range[0]), float(effective_shift_scale_range[1])
            # Sample one shift per intervened node (constant across samples in this dataset).
            sampled_shifts = torch.empty(len(intervention_set), device=self.device, dtype=torch.float).uniform_(low, high)
            shift_term = sampled_shifts.view(1, -1)
            self.last_intervention_shifts = sampled_shifts.detach().cpu().numpy()
        else:
            self.last_intervention_shifts = None

        intervention_scale_term = float(self.intervention_scale)
        if intervention_set[0] is not None and effective_variable_intervention_scale:
            if effective_intervention_scale_range is None:
                upper = abs(float(self.intervention_scale))
                low, high = 0.0, upper
            else:
                low, high = float(effective_intervention_scale_range[0]), float(effective_intervention_scale_range[1])
            sampled_scales = torch.empty(len(intervention_set), device=self.device, dtype=torch.float).uniform_(low, high)
            intervention_scale_term = sampled_scales.view(1, -1)
            self.last_intervention_scales = sampled_scales.detach().cpu().numpy()
        else:
            self.last_intervention_scales = None

        with torch.no_grad():
            observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
            U = torch.zeros(self.n_nodes, self.n_nodes, device=self.device).float()
            U[observed_set, observed_set] = 1
            
            
        if lat_provided:
            E = latent_vec.T
            if not isinstance(E, torch.Tensor):
                # ensure torch tensor on correct device/dtype
                E = torch.tensor(E, device=self.device, dtype=torch.float)
            else:
                E = E.to(device=self.device, dtype=torch.float)
        else:
            E = torch.zeros((n_samples, self.n_nodes), device=self.device, dtype=torch.float)

            if self.noise_type == 'gaussian':
                if len(observed_set) > 0:
                    E[:, observed_set] = self.noise_scale * torch.randn(n_samples, len(observed_set), device=self.device, dtype=torch.float)
                if intervention_set[0] is not None:
                    E[:, intervention_set] = intervention_scale_term * torch.randn(n_samples, len(intervention_set), device=self.device, dtype=torch.float) + shift_term
            elif self.noise_type == 'exponential':
                # torch Exponential uses rate = 1/scale
                if len(observed_set) > 0:
                    dist_obs = torch.distributions.Exponential(rate=1.0 / float(self.noise_scale))
                    E[:, observed_set] = dist_obs.sample((n_samples, len(observed_set))).to(device=self.device, dtype=torch.float)
                if intervention_set[0] is not None:
                    dist_int = torch.distributions.Exponential(rate=1.0)
                    E[:, intervention_set] = dist_int.sample((n_samples, len(intervention_set))).to(device=self.device, dtype=torch.float) * intervention_scale_term + shift_term
            elif self.noise_type == 'gumbel':
                dist_obs = torch.distributions.Gumbel(loc=0.0, scale=float(self.noise_scale))
                if len(observed_set) > 0:
                    E[:, observed_set] = dist_obs.sample((n_samples, len(observed_set))).to(device=self.device, dtype=torch.float)
                if intervention_set[0] is not None:
                    dist_int = torch.distributions.Gumbel(loc=0.0, scale=1.0)
                    E[:, intervention_set] = dist_int.sample((n_samples, len(intervention_set))).to(device=self.device, dtype=torch.float) * intervention_scale_term + shift_term
            elif self.noise_type == 'laplace':
                dist_obs = torch.distributions.Laplace(loc=0.0, scale=float(self.noise_scale))
                if len(observed_set) > 0:
                    E[:, observed_set] = dist_obs.sample((n_samples, len(observed_set))).to(device=self.device, dtype=torch.float)
                if intervention_set[0] is not None:
                    dist_int = torch.distributions.Laplace(loc=0.0, scale=1.0)
                    E[:, intervention_set] = dist_int.sample((n_samples, len(intervention_set))).to(device=self.device, dtype=torch.float) * intervention_scale_term + shift_term
            else:
                raise ValueError(f"Unknown noise_type: {self.noise_type}")


            I = torch.eye(self.n_nodes, device=self.device, dtype=torch.float)
            X = torch.randn(n_samples, self.n_nodes, device=self.device, dtype=torch.float)
            for _ in range(n_iter):
                X = (self.f(X) @ U) + (self.f_i(X) @ (I - U)) + E
        
        if return_latents:
            return X.cpu().numpy(), E.cpu().numpy()
        else:
            return X.detach().cpu().numpy()



class hybridSEM:
    """Hybrid SEM: mixture of linear and tanh-transformed linear effects.

    Interventions follow the same noise-based pattern as other soft intervention
    classes: observed and intervened nodes can use different exogenous noise scales.
    """

    def __init__(
        self,
        graph,
        abs_weight_low=0.2,
        abs_weight_high=0.9,
        noise_scale=0.5,
        contractive=True,
        beta=1.0,
    ):
        self.graph = graph
        self.abs_weight_low = abs_weight_low
        self.abs_weight_high = abs_weight_high
        self.contractive = contractive

        self.n_nodes = len(graph.nodes)
        self.weights = np.random.uniform(self.abs_weight_low, self.abs_weight_high, size=(self.n_nodes, self.n_nodes))
        self.weights *= 2 * np.random.binomial(1, 0.5, size=self.weights.shape) - 1
        self.weights *= nx.to_numpy_array(self.graph)

        self.noise_scale = noise_scale
        self.beta = beta

        if not self.contractive:
            self.weights = make_non_cotractive(self.weights)
        else:
            self.weights = make_contractive(self.weights)

        # Keep a single nonlinear mechanism for all environments.
        self.lip_const = 2.0
        self.n_hidden = 1
        self.bias = False
        self.act_fun = 'tanh'
        self.f = nonlinearMLP(
            n_nodes=self.n_nodes,
            lip_constant=self.lip_const,
            n_layers=self.n_hidden,
            bias=self.bias,
            activation_fn=self.act_fun,
            graph_given=True,
            graph=self.graph,
        )

    def generateData(
        self,
        n_samples,
        intervention_set=[None],
        lat_provided=False,
        latent_vec=None,
        fixed_intervention=False,
        return_latents=False,
        n_iter=30,
        beta_given=False,
        beta=1.0,
        shift_scale=0,
        intervention_scale=0.5,
        noise_type='gaussian',
    ):
        observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
        U = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float)
        U[observed_set, observed_set] = 1.0
        I = torch.eye(self.n_nodes, dtype=torch.float)

        self.intervention_scale = intervention_scale
        self.shift_scale = shift_scale
        self.noise_type = noise_type

        if lat_provided:
            E = latent_vec
            if not isinstance(E, torch.Tensor):
                E = torch.tensor(E, dtype=torch.float)
            else:
                E = E.to(dtype=torch.float)
            # Keep sample-major shape: (n_samples, n_nodes)
            if E.ndim == 2 and E.shape == (self.n_nodes, n_samples):
                E = E.T
        else:
            E_np = np.zeros((n_samples, self.n_nodes), dtype=float)
            if noise_type == 'gaussian':
                if len(observed_set) > 0:
                    E_np[:, observed_set] = self.noise_scale * np.random.randn(n_samples, len(observed_set))
                if intervention_set[0] is not None:
                    E_np[:, intervention_set] = self.intervention_scale * np.random.randn(n_samples, len(intervention_set)) + self.shift_scale
            elif noise_type == 'exponential':
                if len(observed_set) > 0:
                    E_np[:, observed_set] = np.random.exponential(scale=self.noise_scale, size=(n_samples, len(observed_set)))
                if intervention_set[0] is not None:
                    E_np[:, intervention_set] = np.random.exponential(scale=self.intervention_scale, size=(n_samples, len(intervention_set))) + self.shift_scale
            elif noise_type == 'gumbel':
                if len(observed_set) > 0:
                    E_np[:, observed_set] = np.random.gumbel(loc=0.0, scale=self.noise_scale, size=(n_samples, len(observed_set)))
                if intervention_set[0] is not None:
                    E_np[:, intervention_set] = np.random.gumbel(loc=0.0, scale=self.intervention_scale, size=(n_samples, len(intervention_set))) + self.shift_scale
            elif noise_type == 'laplace':
                if len(observed_set) > 0:
                    E_np[:, observed_set] = np.random.laplace(loc=0.0, scale=self.noise_scale, size=(n_samples, len(observed_set)))
                if intervention_set[0] is not None:
                    E_np[:, intervention_set] = np.random.laplace(loc=0.0, scale=self.intervention_scale, size=(n_samples, len(intervention_set))) + self.shift_scale
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            E = torch.tensor(E_np, dtype=torch.float)

        W = torch.tensor(self.weights, dtype=torch.float)
        wtx = lambda x: x @ W

        beta_ = beta if beta_given else self.beta
        beta_ = float(beta_)
        if beta_ < 0.0 or beta_ > 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta_}")

        X = torch.randn(n_samples, self.n_nodes, dtype=torch.float)

        for _ in range(n_iter):
            lin_x = wtx(X)
            nnl_x = self.f(X)
            mix_x = (1.0 - beta_) * lin_x + beta_ * nnl_x
            # Match soft-intervention semantics from nnl: mechanism active on all nodes,
            # while intervention effects are injected via E.
            X = mix_x @ U + mix_x @ (I - U) + E

        if return_latents:
            return X.detach().cpu().numpy(), E.detach().cpu().numpy()
        return X.detach().cpu().numpy()


