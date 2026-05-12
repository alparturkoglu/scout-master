import os
import argparse
import time
import math
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torch.nn.functional as F
import copy
from datagen.generateDataset_softinterventions import Dataset
from datagen.torchDataset import MixedExperimentDataset, experimentDataset, experimentDatasetStrat
from models.functions_softinterventions import indMLPFunction, linearFunction, nonlinearMLP, factorMLPFunction, gumbelSoftMLP
from models.scout_resblock import iResBlock
from models.layers.mlpLipschitz import linearLipschitz
from utils import *

# Helper functions

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def standard_normal_logprob(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales.pow(2)))
    return logZ - z.pow(2) / (2 * (noise_scales.pow(2)))

def computeNLL(latent, intervention_set, logdetgrad, noise_scales):
    observed_set = np.setdiff1d(np.arange(latent.shape[1]), intervention_set)
    logpe = standard_normal_logprob(latent, noise_scales=noise_scales).sum(1, keepdim=True)
    logpx = logpe + logdetgrad
    return -torch.mean(logpx).detach().cpu().numpy()

def compute_loss(model, x, intervention_mask, l1_regularize=False, lambda_c=1e-2, lambda_r=1e-2, fun_type=None, exp_id=None, unknown_intervention=False, learn_interv=True):
    z, logdetgrad, log_det_jacobian_realnvp = model(x, intervention_mask, logdet=True, exp_id=exp_id)
    logpz = standard_normal_logprob(z, noise_scales=torch.ones_like(z)).sum(1, keepdim=True)
    logpe = logpz + log_det_jacobian_realnvp.unsqueeze(1)
    logpx = logpe + logdetgrad        
    loss = -torch.mean(logpx)
    if l1_regularize:
        if fun_type == 'fac-mlp':
            l1_norm = model.f.get_w_adj().abs().sum()
        elif fun_type == 'gst-mlp':
            l1_norm = model.return_adjacency().abs().sum()
        else:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
        if learn_interv and unknown_intervention:
            lr_norm = (model.trained_interv.get_proba()).abs().sum()
            loss += (lambda_r * lr_norm)
        loss += lambda_c * l1_norm

    return loss, torch.mean(logpe), torch.mean(logdetgrad)
    

def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, linearLipschitz):
            m.compute_weight(update=True, n_iterations=n_iterations)

class scout_train_test_wrapper:
    def __init__(self,
                 n_nodes,
                 batch_size=64,
                 l1_reg=False,
                 lambda_c=1e-2,
                 n_lip_iter=5,
                 fun_type='mul-mlp',
                 lip_const=0.9,
                 act_fun='tanh',
                 lr=1e-3,
                 epochs=10,
                 optim='adam',
                 v=False, 
                 inline=False,
                 upd_lip=True,
                 full_input=False, 
                 n_hidden=1, 
                 n_factors=10, 
                 var_o=None,
                 var_i=None,
                 n_power_series=None, 
                 init_var=0.5,
                 lin_logdet=False, 
                 dag_input=False, 
                 thresh_val=1e-2, 
                 centered=True,
                 unknown_intervention=True,
                 lambda_r=1e-2,
                 n_experiments=None,
                 learn_interv=True,
                 tau=0.7,
                 experiment_specific_intervention_noise=False,
                 experiment_specific_intervention_mechanism=False
                 ):

        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.l1_reg = l1_reg
        self.lambda_c = lambda_c
        self.n_lip_iter = n_lip_iter
        self.fun_type = fun_type
        self.lip_const = lip_const
        self.act_fun = act_fun
        self.lr = lr
        self.epochs = epochs
        self.optim = optim
        self.v = v
        self.inline = inline
        self.upd_lip = upd_lip
        self.full_input = full_input
        self.n_hidden = n_hidden
        self.n_factors = n_factors
        self.var_o = var_o
        self.var_i = var_i 
        self.n_power_series = n_power_series
        self.lin_logdet = lin_logdet
        self.thresh_val = thresh_val
        self.centered = centered
        self.unknown_intervention = unknown_intervention
        self.lambda_r = lambda_r
        self.learn_interv = learn_interv
        self.tau = tau
        self.experiment_specific_intervention_noise = experiment_specific_intervention_noise
        self.experiment_specific_intervention_mechanism = experiment_specific_intervention_mechanism
        if n_experiments is None:
            self.n_experiments = n_nodes
        else:
            self.n_experiments = n_experiments  # placeholder, updated during training if needed
        if self.v or self.inline:
            print("Initializing the model")

        if self.fun_type == 'mul-mlp':
            self.f = indMLPFunction(n_nodes=self.n_nodes, lip_constant=self.lip_const, activation=self.act_fun, full_input=self.full_input, n_layers=n_hidden)
            self.f_i = indMLPFunction(n_nodes=self.n_nodes, lip_constant=self.lip_const, activation=self.act_fun, full_input=self.full_input, n_layers=n_hidden)
        elif self.fun_type == 'lin-mlp':
            self.f = linearFunction(n_nodes=self.n_nodes, lip_constant=self.lip_const, full_input=self.full_input)
            self.f_i = linearFunction(n_nodes=self.n_nodes, lip_constant=self.lip_const, full_input=self.full_input)
        elif self.fun_type == 'nnl-mlp':
            self.f = nonlinearMLP(n_nodes=self.n_nodes, lip_constant=self.lip_const, n_layers=self.n_hidden, full_input=self.full_input, activation_fn=self.act_fun)
            self.f_i = nonlinearMLP(n_nodes=self.n_nodes, lip_constant=self.lip_const, n_layers=self.n_hidden, full_input=self.full_input, activation_fn=self.act_fun)
        elif self.fun_type == 'fac-mlp':
            self.f = factorMLPFunction(n_nodes=self.n_nodes, n_factors=self.n_factors, lip_constant=self.lip_const, n_hidden=self.n_hidden, activation=self.act_fun)
            self.f_i = factorMLPFunction(n_nodes=self.n_nodes, n_factors=self.n_factors, lip_constant=self.lip_const, n_hidden=self.n_hidden, activation=self.act_fun)
        elif self.fun_type == 'gst-mlp':
            self.f = gumbelSoftMLP(n_nodes=self.n_nodes, lip_constant=self.lip_const, n_hidden=self.n_hidden, activation=self.act_fun)
            self.f_i = gumbelSoftMLP(n_nodes=self.n_nodes, lip_constant=self.lip_const, n_hidden=self.n_hidden, activation=self.act_fun)


        self.model = iResBlock(self.f,self.f_i, n_power_series=self.n_power_series, var_o=self.var_o, var_i=self.var_i,  init_var=init_var, dag_input=dag_input, lin_logdet=self.lin_logdet, centered=self.centered, total_exp=self.n_experiments, batch_size=self.batch_size, learn_interv=self.learn_interv, tau=self.tau, experiment_specific_intervention_noise=self.experiment_specific_intervention_noise, experiment_specific_intervention_mechanism=self.experiment_specific_intervention_mechanism)
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.model = self.model.to(self.device)
        if self.v or self.inline:
            print("Number of Parameters : {}".format(count_parameters(self.model)))
    
    def n_parameters(self):
        return count_parameters(self.model)
    def train(self, datasets, intervention_sets, return_time=False, return_loss=False, batch_size=64):
        mixed_dataset = MixedExperimentDataset(datasets, intervention_sets)
        mixed_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
     
        if self.inline:
            print("Starting Training")

        if self.optim == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=self.lr)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        # quick diagnostic: ensure intervention logits parameters are present in optimizer
        try:
            if self.inline:
                n_interv_params = sum(1 for n, p in self.model.named_parameters() if 'trained_interv' in n)
                print(f"Intervention logits parameter count: {n_interv_params}")
        except Exception:
            pass


        start_time = time.time()
        self.epoch_times_sec = []
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            for batch_idx, (x_batch, mask_batch, exp_id_batch) in enumerate(mixed_loader):
                if isinstance(x_batch, np.ndarray):
                    x_batch = torch.tensor(x_batch, dtype=torch.float32)
                if isinstance(mask_batch, np.ndarray):
                    mask_batch = torch.tensor(mask_batch, dtype=torch.float32)
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                optimizer.zero_grad()
                # compute loss using the experiment's intervention_set (passed through intervention_sets)
                if self.unknown_intervention == False:
                    exp_intervention_mask = mask_batch
                else:
                    exp_intervention_mask = None
                loss, logpe, logdetgrad = compute_loss(self.model, x_batch, exp_intervention_mask, l1_regularize=self.l1_reg, fun_type=self.fun_type, lambda_c=self.lambda_c, lambda_r=self.lambda_r, exp_id=exp_id_batch, unknown_intervention=self.unknown_intervention, learn_interv=self.learn_interv)

                 
                loss.backward()
                optimizer.step()
                if self.upd_lip:
                    update_lipschitz(self.model, self.n_lip_iter)

                if self.v or self.inline:
                    print(f"Epoch {epoch+1}/{self.epochs} Batch {batch_idx}, Loss {loss.item():.4f}")
            epoch_elapsed = time.time() - epoch_start_time
            self.epoch_times_sec.append(epoch_elapsed)
            if self.v or self.inline:
                em, es = divmod(int(epoch_elapsed), 60)
                eh, em = divmod(em, 60)
                print(f"Epoch {epoch+1}/{self.epochs} time: {eh:d}:{em:02d}:{es:02d} ({epoch_elapsed:.2f}s)")
        stop_time = time.time()
        self.total_training_time_sec = stop_time - start_time
        seconds = int(stop_time - start_time)
        if self.v or self.inline:
            m_all, s_all = divmod(seconds, 60)
            h_all, m_all = divmod(m_all, 60)
            print(f"Overall training time: {h_all:d}:{m_all:02d}:{s_all:02d} ({self.total_training_time_sec:.2f}s)")
        
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if return_time and return_loss:
            return (h, m, s), (loss.item(), logdetgrad.item())
        elif return_time:
            return h, m, s
        elif return_loss:
            return loss.item(), logdetgrad.item()
        
    def threshold(self):
        # compute adjacency, binarize and freeze masks
        W = self.get_adjacency()
        adj_mat = (W >= self.thresh_val)

        # set graph_given and graph_adj on both f and f_i so their forward respects frozen graph
        try:
            self.model.f.graph_given = True
            self.model.f.graph_adj = adj_mat
        except Exception:
            pass
        try:
            self.model.f_i.graph_given = True
            self.model.f_i.graph_adj = adj_mat
        except Exception:
            pass

        # if iResBlock owns the GumbelAdjacency sampler, tell it to freeze (if API exists)
        try:
            if hasattr(self.model, 'gumbel_soft_layer') and hasattr(self.model.gumbel_soft_layer, 'freeze_threshold'):
                self.model.gumbel_soft_layer.freeze_threshold(self.thresh_val)
        except Exception:
            pass

    def get_adjacency(self):
        # return a numeric adjacency matrix (probabilities) for different function types
        if self.fun_type == 'fac-mlp':
            W = np.abs(self.f.get_w_adj().detach().cpu().numpy())
        elif self.fun_type == 'gst-mlp':
            # adjacency exposed by iResBlock (Gumbel layer)
            try:
                W = np.abs(self.model.return_adjacency().detach().cpu().numpy())
            except Exception:
                # fallback to f if available
                W = np.abs(getattr(self.f, 'get_w_adj', lambda: torch.zeros((self.n_nodes, self.n_nodes)))().detach().cpu().numpy())
        else:
            # try helper that extracts adj from a single function; utils should provide get_adj_from_single_func
            try:
                W = get_adj_from_single_func(self.f, device=self.device)
            except Exception:
                # last resort: zeros
                W = np.zeros((self.n_nodes, self.n_nodes))

        # if user provided graph mask, apply it
        if getattr(self.model.f, 'graph_given', False):
            return (self.model.f.graph_adj * W).astype(float)

        return W
    
    def get_auroc(self, W_gt):
        _, _, area = compute_auroc(W_gt, self.get_adjacency())
        return area

    def get_shd(self, W_gt):
        W_est = self.model.f.graph_adj
        shd, _ = compute_shd(W_gt, W_est)
        return shd

    def get_auprc(self, W_gt, n_points=50):
        baseline, area = compute_auprc(W_gt, self.get_adjacency(), n_points=n_points)
        return baseline, area

    def store_figure(self, graph, generative_model, output_path="figures", gid=1):
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True, ax=axs[0])
        axs[0].set_title("Graph")

        axs[1].set_title("Ground Truth - Adj")
        axs[1].imshow(np.abs(generative_model.weights) > 0)

        W = self.get_adjacency()
        axs[2].set_title("Estimated - Adj")
        axs[2].imshow(W)
        plt.savefig(os.path.join(output_path, 'd_{}_g_{}_f_{}_af_{}.png'.format(self.n_nodes, gid, self.fun_type, self.act_fun)))

    def predict(self, latents, intervention_sets, n_iter=10, init_provided=False, x_init=None):
        pred_datasets = list()
        i = 0
        for latent, intervention_set in zip(latents, intervention_sets):
            lat_t = torch.tensor(latent).float().to(self.model.device)
            data_pred = self.model.predict_from_latent(lat_t, n_iter, intervention_set=intervention_set, init_provided=init_provided, x_init=x_init[i]) 
            i += 1
            data_pred = data_pred.detach().cpu().numpy()
            pred_datasets.append(data_pred)
        return pred_datasets    
    
    def forwardPass(self, datasets):
        predictions = list()
        for dataset in datasets:
            data_t = torch.tensor(dataset).float().to(self.device)
            f_x = self.model.f(data_t, self.model.gumbel_soft_layer(data_t.shape[0]))
            predictions.append(f_x.detach().cpu().numpy())
        
        return predictions
    
    def predictLikelihood(self, datasets, intervention_sets):
        likelihood_list = list()
        for dataset, intervention_set in zip(datasets, intervention_sets):
            data_t = torch.tensor(dataset).float().to(self.device)
            if intervention_set is None or (isinstance(intervention_set, (list, tuple)) and len(intervention_set) == 0) or (isinstance(intervention_set, list) and intervention_set[0] is None):
                intervention_mask = torch.ones_like(data_t)
            else:
                intervention_mask = torch.ones_like(data_t)
                intervention_mask[:, intervention_set] = 0
            latents, logdetgrad, log_det_jacobian_realnvp = self.model(data_t, intervention_mask, logdet=True, neumann_grad=False)
            logpz = standard_normal_logprob(latents, noise_scales=torch.ones_like(latents)).sum(1, keepdim=True)
            logpe = logpz + log_det_jacobian_realnvp.unsqueeze(1)
            logpx = logpe + logdetgrad
            nll = -torch.mean(logpx).detach().cpu().numpy()
            likelihood_list.append(nll.item()/self.n_nodes)
        
        return likelihood_list
    
    def predictSamples(self, intervention_set, n_samples=5000, x_init=None):
        noise_scale = np.diag(np.exp(self.model.var_o.detach().cpu().numpy()))
        latent_vec = np.random.randn(n_samples, self.n_nodes) @ noise_scale
        latent_vec = torch.tensor(latent_vec).float().to(self.device)
        x_init = torch.tensor(x_init).float().to(self.device)
        x = self.model.predict_from_latent(latent_vec, intervention_set=intervention_set, x_init=x_init)
        return x  

    def _pred_cond_mean_for_intervention(self, dataset, intervention_set):
        noise_scale = np.diag(np.exp(self.model.var_o.detach().cpu().numpy()))
        latent_vec = np.random.randn(dataset.shape[0], dataset.shape[1]) @ noise_scale
        latent_vec = torch.tensor(latent_vec).float().to(self.device)
        x_init = torch.tensor(dataset).float().to(self.device)
        x = self.model.predict_from_latent(latent_vec, intervention_set=intervention_set, x_init=x_init)
        return x.detach().cpu().numpy().mean(axis=0)

    def predictConditionalMean(self, datasets, intervention_sets):
        cond_mean_list = list()
        for dataset, intervention_set in zip(datasets, intervention_sets):
            cond_mean = self._pred_cond_mean_for_intervention(dataset, intervention_set)
            cond_mean_list.append(cond_mean)
        return cond_mean_list

if __name__ == '__main__':

    # Parsing command line arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_nodes', type=int, default=5)
    parser.add_argument('--exp_dens', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--gen_model', type=str, choices=['lin', 'nnl'], default='lin')
    parser.add_argument('--n_exp', type=int, default=5)
    parser.add_argument('--mode', type=str, choices=['indiv-node', 'no-constraint', 'sat-pair-condition'], default='indiv-node')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l1_reg', action='store_true', default=False)
    parser.add_argument('--lambda_c', type=float, default=1e-2)
    parser.add_argument('--n_lip_iter', type=int, default=5)
    parser.add_argument('--fun_type', type=str, choices=['mul-mlp', 'lin-mlp', 'nnl-mlp'], default='mul-mlp')
    parser.add_argument('--lip_const', type=float, default=0.9)
    parser.add_argument('--act_fun', type=str, choices=['tanh', 'relu', 'sigmoid'], default='tanh')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--cosine-T0', type=int, default=20)
    parser.add_argument('--cosine-Tmult', type=int, default=2)
    parser.add_argument('--gid', type=int, default=1)
    parser.add_argument('--v', action='store_true', default=False)
    parser.add_argument('--store_fig', action='store_true', default=False)
    parser.add_argument('--inline', action='store_true', default=False)
    parser.add_argument('--upd_lip', action='store_true', default=False)
    parser.add_argument('--full_input', action='store_true', default=False)
    parser.add_argument('--dag-input', action='store_true', default=False)
    parser.add_argument('--no-inter', action='store_false', default=True) 
    parser.add_argument('--lin-logdet', action='store_true', default=False)
    
    args = parser.parse_args()



    # Generate the Graph and the dataset. 

    print("Generating the graph and the dataset")

    dataset_gen = Dataset(n_nodes=args.n_nodes, 
                        expected_density=args.exp_dens, 
                        n_samples=args.n_samples, 
                        n_experiments=args.n_exp, 
                        mode=args.mode, 
                        enforce_dag=True)
    dataset = dataset_gen.generate()
    graph = dataset_gen.graph
    generative_model = dataset_gen.gen_model


    resblock = resflow_train_test_wrapper(n_nodes=args.n_nodes,
                               batch_size=args.batch_size,
                               l1_reg=args.l1_reg,
                               lambda_c=args.lambda_c,
                               n_lip_iter=args.n_lip_iter,
                               fun_type=args.fun_type,
                               lip_const=args.lip_const,
                               act_fun=args.act_fun,
                               lr=args.lr,
                               epochs=args.epochs, 
                               optim=args.optim,
                               v=args.v,
                               inline=args.inline,
                               upd_lip=args.upd_lip,
                               full_input=args.full_input,
                               warmup_iters=args.warmup_iters,
                               cosine_T0=args.cosine_T0,
                               cosine_Tmult=args.cosine_Tmult,
                               lin_logdet=args.lin_logdet,
                               dag_input=args.dag_input)
    h, m, s = resblock.train(dataset, dataset_gen.targets, return_time=True, batch_size=args.batch_size)

    if args.store_fig:
        resblock.store_figure(graph, generative_model, gid=args.gid)

    area = resblock.get_auprc(np.abs(generative_model.weights) > 0)
    print()
    print("ID: {}, Elapsed time: {:d}:{:02d}:{:02d}, AUPRC: {}".format(args.gid, h, m, s, area))
    lat_var = np.exp(resblock.model.var_o.detach().cpu().numpy())
    print("Estimated Latent variance: {}".format(lat_var))

    val_dataset_gen = Dataset(n_nodes=args.n_nodes,
                             expected_density=1, 
                             n_samples=1000, 
                             n_experiments=10, 
                             mode='no-constraint',
                             min_targets=2,
                             max_targets=2,
                             graph_provided=True,
                             graph=graph,
                             gen_model_provided=True,
                             gen_model=generative_model)
    val_datasets = val_dataset_gen.generate(fixed_interventions=True)
    nll_list = resblock.predictLikelihood(val_datasets, val_dataset_gen.targets)
    print("Average NLL: {}".format(np.mean(nll_list)))