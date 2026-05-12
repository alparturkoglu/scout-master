import torch
import torch.nn as nn 
import numpy as np 
import math 
import time 
import copy
from models.layers.masks import GumbelAdjacency, GumbelIntervWeight
from models.mynflows import MyNFlows
# per-node Bernoulli logits used; no categorical import needed


class iResBlock(nn.Module):
    """
    ----------------------------------------------------------------------------------------
    The class for a single residual map, i.e., (I -f)(x) = e. 
    ----------------------------------------------------------------------------------------
    The forward method computes the residual map and also log-det-Jacobian of the map. 

    Parameters:
    1) func - (nn.Module) - torch module for modelling the function f in (I - f).
    2) n_power_series - (int/None) - Number of terms used for computing determinent of log-det-Jac, 
                                     set it to None to use Russian roulette estimator. 
    3) neumann_grad - (bool) - If True, Neumann gradient estimator is used for Jacobian.
    4) n_dist - (string) - distribution used to sample n when using Russian roulette estimator. 
                           'geometric' - geometric distribution.
                           'poisson' - poisson distribution.
    5) lamb - (float) - parameter of poisson distribution.
    6) geom_p - (float) - parameter of geometric distribution.
    7) n_samples - (int) - number of samples to be sampled from n_dist. 
    8) grad_in_forward - (bool) - If True, it will store the gradients of Jacobian with respect to 
                                  parameters in the forward pass. 
    9) n_exact_terms - (int) - Minimum number of terms in the power series. 
    """
    def __init__(self, func, func_i, n_power_series, neumann_grad=True, n_dist='geometric', lamb=2., geom_p=0.5, n_samples=1, grad_in_forward=False, n_exact_terms=2, var_o=None, var_i=None , init_var=0.5, dag_input=False, lin_logdet=False, centered=True, total_exp=None, batch_size=128, learn_interv=True, tau=0.7, experiment_specific_intervention_noise=True, experiment_specific_intervention_mechanism=False):
        super(iResBlock, self).__init__()
        self.f = func
        self.f_i = func_i
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_dist = n_dist
        self.n_power_series = n_power_series 
        self.neumann_grad = neumann_grad 
        self.grad_in_forward = grad_in_forward
        self.n_exact_terms = n_exact_terms
        self.n_samples = n_samples
        self.dag_input = dag_input
        self.gumbel_soft_layer = GumbelAdjacency(self.f.n_nodes)
        self.lin_logdet = lin_logdet
        self.centered = centered
        self.total_exp = total_exp
        self.batch_size = batch_size
        self.learn_interv = learn_interv 
        self.tau = tau # global flag to control whether to learn intervention masks
        self.experiment_specific_intervention_noise = experiment_specific_intervention_noise
        self.experiment_specific_intervention_mechanism = experiment_specific_intervention_mechanism

        self.mu = nn.Parameter(torch.zeros(self.f.n_nodes).float())

        if dag_input:
            self.Lambda = nn.Parameter(torch.zeros(self.f.n_nodes).float())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Pre-create RealNVP flows for every possible marginal dimension (1..n_nodes)
        # so we can handle arbitrary subsets of intervened / observed dimensions.
        self.realnvp_by_node_obs = nn.ModuleDict()
        for d in range(1, self.f.n_nodes + 1):
            self.realnvp_by_node_obs[str(d)] = MyNFlows(1).to(device)
        self.realnvp_by_node_int = nn.ModuleDict()
        if not self.experiment_specific_intervention_noise:
            # Old behavior: one intervention flow per node, shared across experiments.
            for d in range(1, self.f.n_nodes + 1):
                self.realnvp_by_node_int[str(d)] = MyNFlows(1).to(device)
        # New behavior: different intervention noise transforms per experiment.
        elif self.total_exp is not None and int(self.total_exp) > 0:
            for exp_idx in range(int(self.total_exp)):
                per_exp_flows = nn.ModuleDict()
                for d in range(1, self.f.n_nodes + 1):
                    per_exp_flows[str(d)] = MyNFlows(1).to(device)
                self.realnvp_by_node_int[str(exp_idx)] = per_exp_flows
        else:
            # Backward-compatible shared fallback when experiment id is unavailable.
            shared_flows = nn.ModuleDict()
            for d in range(1, self.f.n_nodes + 1):
                shared_flows[str(d)] = MyNFlows(1).to(device)
            self.realnvp_by_node_int['shared'] = shared_flows

        # Optional: learn different intervention mechanisms f_i per experiment.
        self.f_i_by_exp = nn.ModuleDict()
        if self.experiment_specific_intervention_mechanism and self.total_exp is not None and int(self.total_exp) > 0:
            for exp_idx in range(int(self.total_exp)):
                self.f_i_by_exp[str(exp_idx)] = copy.deepcopy(self.f_i).to(device)

        self.trained_interv = GumbelIntervWeight(self.f.n_nodes, self.total_exp, tau=self.tau)
        self.last_interv_mask = None  # torch.Tensor of shape (n_nodes,)
        self.last_interv_indices = None  # python list of ints (or None)

    def _resolve_exp_ids(self, exp_id, batch_size, device):
        """Return per-sample experiment ids as LongTensor of shape (B,)."""
        if exp_id is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if isinstance(exp_id, torch.Tensor):
            exp_tensor = exp_id.to(device=device, dtype=torch.long).view(-1)
        elif isinstance(exp_id, (list, tuple, np.ndarray)):
            exp_tensor = torch.as_tensor(exp_id, dtype=torch.long, device=device).view(-1)
        else:
            exp_tensor = torch.tensor([int(exp_id)], dtype=torch.long, device=device)

        if exp_tensor.numel() == 1:
            return exp_tensor.repeat(batch_size)
        if exp_tensor.numel() != batch_size:
            raise ValueError(f"exp_id has {exp_tensor.numel()} elements, expected 1 or {batch_size}")
        return exp_tensor

    def _get_intervention_flow(self, exp_idx, node_idx):
        """Pick intervention flow for (experiment, node), creating per-exp flows lazily if needed."""
        exp_key = str(int(exp_idx))
        node_key = str(int(node_idx))

        # Old behavior path: shared intervention flow per node for all experiments.
        if not self.experiment_specific_intervention_noise:
            return self.realnvp_by_node_int[node_key]

        # Legacy/shared setup.
        if 'shared' in self.realnvp_by_node_int:
            return self.realnvp_by_node_int['shared'][node_key]

        # Lazily support unseen experiment ids.
        if exp_key not in self.realnvp_by_node_int:
            target_device = next(self.parameters()).device
            per_exp_flows = nn.ModuleDict()
            for d in range(1, self.f.n_nodes + 1):
                per_exp_flows[str(d)] = MyNFlows(1).to(target_device)
            self.realnvp_by_node_int[exp_key] = per_exp_flows

        return self.realnvp_by_node_int[exp_key][node_key]

    def _eval_func(self, func_module, x, graph_adj=None):
        """Evaluate module while supporting both signatures: f(x) and f(x, graph_adj)."""
        if graph_adj is None:
            return func_module(x)
        try:
            return func_module(x, graph_adj)
        except TypeError:
            return func_module(x)

    def _get_f_i_for_exp(self, exp_idx):
        """Get per-experiment f_i module, creating it lazily if needed."""
        if not self.experiment_specific_intervention_mechanism:
            return self.f_i

        exp_key = str(int(exp_idx))
        if exp_key not in self.f_i_by_exp:
            target_device = next(self.parameters()).device
            self.f_i_by_exp[exp_key] = copy.deepcopy(self.f_i).to(target_device)
        return self.f_i_by_exp[exp_key]

    def _compute_f_i_batch(self, x, exp_ids=None, graph_adj=None):
        """Compute f_i(x) either shared or experiment-specific (grouped by exp_ids)."""
        if not self.experiment_specific_intervention_mechanism:
            return self._eval_func(self.f_i, x, graph_adj)

        if exp_ids is None:
            exp_ids = self._resolve_exp_ids(None, x.shape[0], x.device)

        f_x_i = torch.zeros_like(x)
        for exp_val in torch.unique(exp_ids):
            exp_mask = (exp_ids == exp_val).nonzero(as_tuple=True)[0]
            x_exp = x.index_select(0, exp_mask)
            g_exp = None
            if isinstance(graph_adj, torch.Tensor) and graph_adj.dim() >= 3 and graph_adj.shape[0] == x.shape[0]:
                g_exp = graph_adj.index_select(0, exp_mask)
            f_i_module = self._get_f_i_for_exp(exp_val.item())
            f_x_i_exp = self._eval_func(f_i_module, x_exp, g_exp)
            f_x_i[exp_mask] = f_x_i_exp
        return f_x_i

    def forward(self, x, intervention_mask=None, logdet=False, neumann_grad=True, logdet_time_measure=False, exp_id=None):
        # set intervention set to [None]
        self.neumann_grad = neumann_grad        
        
        if intervention_mask is None and self.learn_interv == True:
            B = x.shape[0]
            # Build regime indices: either per-sample (one index per batch row) or single index repeated
            # Here we create a per-batch repeated index for the given exp_id (common case)
            # Ensure regime_idx is on the same device as x
            if isinstance(exp_id, torch.Tensor):
                regime_idx = exp_id.clone().detach().to(device=x.device, dtype=torch.long)
            else:
                regime_idx = torch.tensor(exp_id, dtype=torch.long, device=x.device)  # scalar
            regime_for_call_cpu = regime_idx.cpu()

            intervention_mask = self.trained_interv(1, regime=regime_for_call_cpu)
            intervention_mask = 1- intervention_mask.to(x.device).type_as(x)
        elif intervention_mask is None and self.learn_interv == False:
            intervention_mask = torch.ones(x.shape[0], x.shape[1], device=x.device)

        I = torch.ones(x.shape[0], x.shape[1], device=x.device)
        
        if not logdet:
            exp_ids_fi = None
            if self.experiment_specific_intervention_mechanism:
                exp_ids_fi = self._resolve_exp_ids(exp_id, x.shape[0], x.device)
            f_x = self._eval_func(self.f, x)
            f_x_i = self._compute_f_i_batch(x, exp_ids=exp_ids_fi)
            y = x - f_x * intervention_mask - f_x_i * (I - intervention_mask)
            return y
        else:
            if self.dag_input:
                Lamb_mat = torch.diag(torch.exp(self.Lambda))
                Lamb_mat_inv = torch.diag(1/torch.exp(self.Lambda))
                x_inp = (x - self.mu) @ Lamb_mat
            else:
                x_inp = x
            f_x, f_x_i, logdetgrad, cmp_time = self._logdetgrad(x_inp, intervention_mask, exp_id=exp_id)
            if logdet_time_measure:
                return x - f_x * intervention_mask - f_x_i * (I - intervention_mask), logdetgrad, cmp_time
            else:
                if self.dag_input:
                    e = (x - self.mu) - (f_x* intervention_mask) @ Lamb_mat_inv  - (f_x_i * (1 - intervention_mask)) @ Lamb_mat_inv 
                else:
                    #e = (x - self.mu) - f_x @ U - f_x_i @ (I - U) 
                    e = x - f_x * intervention_mask - f_x_i * (I - intervention_mask)
            e = e.to(self.device) 

            # assume:
            # e: Tensor shape (B, D)  -- residuals e computed earlier
            # intervention_mask: Tensor shape (B, D) with 1.0 for observed, 0.0 for intervened
        B, D = e.shape
        exp_ids = None
        if self.experiment_specific_intervention_noise:
            exp_ids = self._resolve_exp_ids(exp_id, B, e.device)
    # add a small per-coordinate offset (use torch.linspace so device/dtype match)
        #e = e + 1e-1
        # prepare full-size containers to place per-node outputs back into (keeps alignment)
        z_obs_full = torch.zeros((B, D), device=e.device, dtype=e.dtype)
        z_int_full = torch.zeros((B, D), device=e.device, dtype=e.dtype)
        logdet_obs_per_node = torch.zeros((B, D), device=e.device, dtype=e.dtype)
        logdet_int_per_node = torch.zeros((B, D), device=e.device, dtype=e.dtype)

        # iterate nodes, select non-zero entries per node, call the corresponding d=1 flows,
        # and scatter the results back to the full (B,D) tensors. Skip nodes with no samples.
        for j in range(D):
            col = e[:, j]                          # (B,)
            mask_col = intervention_mask[:, j]    # (B,)

            # indices for observed / intervened in this batch
            obs_idx = (mask_col > 0).nonzero(as_tuple=True)[0]
            int_idx = (mask_col <= 0).nonzero(as_tuple=True)[0]

            # observed
            if obs_idx.numel() > 0:
                obs_col = col.index_select(0, obs_idx).reshape(-1, 1)   # (n_obs_j, 1)
                flow_obs = self.realnvp_by_node_obs[str(j+1)]
                z_o, ld_o = flow_obs(obs_col)
                # place latent back into full-sized tensor at correct rows
                z_obs_full[obs_idx, j] = z_o.reshape(-1)
                logdet_obs_per_node[obs_idx, j] = ld_o.reshape(-1)

            # intervened
            if int_idx.numel() > 0:
                if not self.experiment_specific_intervention_noise:
                    # Fast path: original behavior, one shared intervention flow per node.
                    int_col = col.index_select(0, int_idx).reshape(-1, 1)
                    flow_int = self.realnvp_by_node_int[str(j + 1)]
                    z_i, ld_i = flow_int(int_col)
                    z_int_full[int_idx, j] = z_i.reshape(-1)
                    logdet_int_per_node[int_idx, j] = ld_i.reshape(-1)
                else:
                    int_exp_ids = exp_ids.index_select(0, int_idx)
                    for exp_val in torch.unique(int_exp_ids):
                        exp_mask = (int_exp_ids == exp_val).nonzero(as_tuple=True)[0]
                        exp_int_idx = int_idx.index_select(0, exp_mask)
                        int_col = col.index_select(0, exp_int_idx).reshape(-1, 1)  # (n_int_j_exp, 1)
                        flow_int = self._get_intervention_flow(exp_val.item(), j + 1)
                        z_i, ld_i = flow_int(int_col)
                        z_int_full[exp_int_idx, j] = z_i.reshape(-1)
                        logdet_int_per_node[exp_int_idx, j] = ld_i.reshape(-1)

        # combine per-coordinate latents using the mask
        z_full = z_obs_full * intervention_mask + z_int_full * (1.0 - intervention_mask)

        # per-sample scalar logdet (sum over nodes)
        logdet_per_sample = logdet_obs_per_node.sum(dim=1) + logdet_int_per_node.sum(dim=1)

        return z_full, logdetgrad, logdet_per_sample
    def return_adjacency(self):
        return self.gumbel_soft_layer.get_proba() 

    # TODO have to update this for when self.dag_input = True (DONE - no change needed)
    def predict_from_latent(self, latent_vec, n_iter=10, intervention_set=[None], init_provided=False, x_init=None):
        if init_provided:
            x = torch.tensor(x_init).float().to(latent_vec.device) 
        else:
            x = torch.randn(latent_vec.size(), device=latent_vec.device)
        c = torch.zeros_like(x)
        obs_set = np.setdiff1d(np.arange(x.shape[1]), intervention_set)
        U = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        U[obs_set, obs_set] = 1
        if intervention_set[0] != None:
            c[:, intervention_set] = torch.tensor(x_init[:, intervention_set]).float().to(latent_vec.device)

        for _ in range(n_iter):
            x = self.f(x - self.mu) @ U + latent_vec @ U + c + self.mu
        
        return x 

    def _logdetgrad(self, x, intervention_mask, exp_id=None):
        with torch.enable_grad():
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)
            
            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.

            vareps = torch.randn_like(x)

            if self.lin_logdet:
                estimator_fn = linear_logdet_estimator
            else:
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

            if self.training and self.grad_in_forward:
                f_x, logdetgrad = mem_eff_wrapper(
                    estimator_fn, self.f, x, n_power_series, vareps, coeff_fn, self.training
                )
            else:
                x = x.requires_grad_(True)
                graph_adj = self.gumbel_soft_layer(x.shape[0])
                f_x = self._eval_func(self.f, x, graph_adj)
                exp_ids_fi = None
                if self.experiment_specific_intervention_mechanism:
                    exp_ids_fi = self._resolve_exp_ids(exp_id, x.shape[0], x.device)
                f_x_i = self._compute_f_i_batch(x, exp_ids=exp_ids_fi, graph_adj=graph_adj)
                tic = time.time()
                if self.lin_logdet:
                    Weight = self.f.layer.weight
                    if self.experiment_specific_intervention_mechanism and exp_ids_fi is not None:
                        Weight_i = torch.zeros_like(Weight)
                        for exp_val in torch.unique(exp_ids_fi):
                            exp_mask = (exp_ids_fi == exp_val)
                            frac = exp_mask.float().mean()
                            Weight_i = Weight_i + frac * self._get_f_i_for_exp(exp_val.item()).layer.weight
                    else:
                        Weight_i = self.f_i.layer.weight
                    I = torch.eye(Weight.shape[0], Weight.shape[1], device=Weight.device)
                    # Use the average intervention pattern across the batch for linear mask mixing.
                    U = torch.diag(intervention_mask.float().mean(dim=0)).to(Weight.device)
                    self_loop_mask = torch.ones_like(Weight)
                    ind = np.diag_indices(Weight.shape[0])
                    self_loop_mask[ind[0], ind[1]] = 0
                    logdetgrad = estimator_fn((U @ self_loop_mask * Weight) + ((I-U) @ self_loop_mask * Weight_i), x.shape[0])
                else:
                    I = torch.ones(x.shape[0], x.shape[1], device=intervention_mask.device)
                    #I = torch.eye(x.shape[1], x.shape[1], device=x.device)
                    logdetgrad = estimator_fn((f_x * intervention_mask) + (f_x_i * (I - intervention_mask)), x, n_power_series, vareps, coeff_fn, self.training)
                toc = time.time()
                comp_time = toc - tic 

        return f_x, f_x_i, logdetgrad.view(-1, 1), comp_time
        
def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = -1 / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1) * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad

def linear_logdet_estimator(W, bs):
    n = W.shape[0]
    I = torch.eye(n, device=W.device)
    return torch.log(torch.det(I - W)) * torch.ones(bs, 1, device=W.device)



def mem_eff_wrapper(): # Function to store the gradients in the forward pass. To be implemented. 
    return 0

def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)

def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)

def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)

def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum