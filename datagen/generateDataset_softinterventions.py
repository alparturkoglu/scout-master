import os 
import csv 
import numpy as np
from tqdm import tqdm 
import networkx as nx

from utils import *
from datagen.graph import DirectedGraphGenerator
from datagen.structuralModels_softinterventions import linearSEM, nonlinearSEM, hybridSEM

import argparse

class Dataset:
    """
    -------------------------------------------------------------------------------------------------
    A class that stores the dataset parameters and generates the dataset.
    -------------------------------------------------------------------------------------------------
    Parameters:
    1)  n_nodes - (int) - number of nodes in the graph. 
    2)  expected_density - (int) - expected number of outgoing edges per node.
    3)  n_samples - (list of numbers) - number of samples in each experiment.
    4)  n_experiments - (int) - number of experiments to be performed. 
    5)  target_predef - (bool) - If True, then the target for each experiment has
                                 to be provided by the user. 
                                 Else, the targets are randomly selected in each 
                                 experiment. 
    6)  min_targets - (int) - minimum targets in each experiment.
    7)  max_targets - (int) - maximum targets in each experiment. 
    8)  mode - (string) - 'sat-pair-condition' - the targets are chosen such that
                                                 pair condition is satisfied.
                          'indiv-node' - Each experiment intervenes on a single node.
                                         Note that the pair condition is always satisfied 
                                         in this case. 
                          'no-constraint' - targets are chosen randomly in each experiment
                                            with no further constraints.
    9)  abs_weight_low - (frac) - absolute least value of the edge weights. 
    10) abs_weight_high - (frac) - absolute largest value of the edge weights.  
    11) targets - (list(list)) - list of targets for each experiments, None if target_predef=False.
    12) sem_type - (string) - 'linear' - sample from linear SEM.
                              'non-linear' - sample from non-linear SEM. 
    13) graph_provided - (bool) - True, if the graph is provided as an input.
    14) graph - (nx graph) - the graph definition the parent-child relations.
    15) gen_model_provided - (bool) - True, if the generative model is provided.
    16) gen_model - (gen model instance) - instance of the generative model (linearSEM/nonlinearSEM)
    17) noise_scale - (float) - scale of the noise in the SEM.
    18) n_hidden - (int) - number of hidden layers in the non-linear SEM. Ignored if sem_type is 'lin'.
    19) act_fun - (string) - activation function for the non-linear SEM. Ignored if sem_type is 'lin'.
    20) lip_constant - (float) - Lipschitz constant for the non-linear SEM. Ignored if sem_type is 'lin'.
    21) enforce_dag - (bool) - whether to enforce the graph to be a DAG. If False, the graph can have cycles.
    22) contractive - (bool) - whether to ensure that the generated SEM is contractive. If True, the spectral norm of the weight matrix is less than 1.
    23) shift_scale - (float) - scale of the shift in the soft interventions. Ignored if sem_type is 'lin'.
    24) variable_shift_scale - (bool) - whether to use variable shift scale for each intervention. If True, shift_scale is ignored and shift_scale_range is used to sample the shift scale for each intervention.
    25) shift_scale_low - (float) - lower bound for the shift scale if variable_shift_scale is True.    
    26) shift_scale_high - (float) - upper bound for the shift scale if variable_shift_scale is True.
    27) intervention_scale - (float) - scale of the intervention effect for soft interventions. Ignored if sem_type is 'lin'.
    28) variable_intervention_scale - (bool) - whether to use variable intervention scale for each intervention. If True, intervention_scale is ignored and intervention_scale_range is used to sample the intervention scale for each intervention.
    29) intervention_scale_low - (float) - lower bound for the intervention scale if variable_intervention_scale is True.
    30) intervention_scale_high - (float) - upper bound for the intervention scale if variable_intervention_scale is True.
    31) noisy - (bool) - whether to add noise to the interventions. If True, noise is added to the interventions with scale given by noise_scale.
    32) noise_type - (string) - type of noise to add 
    Here, targets refer to the set of nodes intervened in an experiment. If the number of experiments
    is not sufficient to satisfy the requirement of the constraint, then the value is adjusted to 
    allow for the mode constraint to be satisfied. 
    """

    def __init__(
        self,
        n_nodes,
        expected_density,
        n_samples,
        n_experiments,
        target_predef=False, 
        min_targets=1,
        max_targets=4,
        mode='indiv-node',
        abs_weight_low=0.2,
        abs_weight_high=0.8,
        lip_constant=0.9,
        targets=None,
        sem_type='lin',
        graph_provided=False, 
        graph=None,
        gen_model_provided=False, 
        gen_model=None,
        noise_scale=0.5,
        n_hidden=1, 
        act_fun='tanh', 
        enforce_dag=False,
        contractive=True,
        shift_scale=0,
        variable_shift_scale=False,
        shift_scale_low=None,
        shift_scale_high=None,
        intervention_scale=0.5,
        variable_intervention_scale=False,
        intervention_scale_low=None,
        intervention_scale_high=None,
        noisy=False,
        noise_type='gaussian',
        noisy_weight=-1,
        beta=1
    ):
        self.n_nodes = n_nodes
        self.expected_density = expected_density
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.target_predef = target_predef
        self.min_targets = min_targets
        self.max_targets = max_targets
        self.mode = mode
        self.abs_weight_low = abs_weight_low
        self.abs_weight_high = abs_weight_high
        self.sem_type = sem_type
        self.lip_constant = lip_constant
        self.noise_scale = noise_scale
        self.n_hidden = n_hidden
        self.act_fun = act_fun
        self.enforce_dag = enforce_dag
        self.contractive = contractive
        self.shift_scale = shift_scale
        self.variable_shift_scale = variable_shift_scale
        if shift_scale_low is not None and shift_scale_high is not None:
            self.shift_scale_range = (shift_scale_low, shift_scale_high)
        else:
            self.shift_scale_range = None
        self.variable_intervention_scale = variable_intervention_scale
        if intervention_scale_low is not None and intervention_scale_high is not None:
            self.intervention_scale_range = (intervention_scale_low, intervention_scale_high)
        else:
            self.intervention_scale_range = None
        self.noisy = noisy
        self.noise_type = noise_type
        self.noisy_weight = noisy_weight
        self.beta = beta
        # Scale of intervention effect for soft interventions
        self.intervention_scale = intervention_scale
        self.pair_condition_matrix = np.zeros((self.n_nodes, self.n_nodes)) # intervention (rows) X observations (column)

        if self.target_predef:
            # self.targets stores the set of nodes intervened in each experiment.
            self.targets = targets
            assert len(self.targets) == self.n_experiments, f"Expected len(targets) to be {n_experiments}, got {len(targets)}" 
            self.checkPairCondition()
        
        else:
            self._pick_targets()

        if graph_provided:
            self.graph = graph
        else:
            generator = DirectedGraphGenerator(nodes=self.n_nodes, expected_density=self.expected_density, enforce_dag=self.enforce_dag)
            self.graph = generator()
        
        if gen_model_provided:
            self.gen_model = gen_model
            if not graph_provided:
                self.graph = self.gen_model.graph
        else:
            if self.sem_type == 'lin':
                self.gen_model = linearSEM(self.graph, self.abs_weight_low, self.abs_weight_high, noise_scale=self.noise_scale, contractive=contractive, noisy=self.noisy, noisy_weight=self.noisy_weight)
            elif self.sem_type == 'nnl':
                self.gen_model = nonlinearSEM(
                    self.graph,
                    lip_const=self.lip_constant,
                    noise_scale=self.noise_scale,
                    n_hidden=self.n_hidden,
                    act_fun=self.act_fun,
                    noisy=self.noisy,
                    noisy_weight=self.noisy_weight,
                    contractive=contractive,
                    variable_shift_scale=self.variable_shift_scale,
                    shift_scale_range=self.shift_scale_range,
                    variable_intervention_scale=self.variable_intervention_scale,
                    intervention_scale_range=self.intervention_scale_range,
                )
            elif self.sem_type == 'hyb':
                self.gen_model = hybridSEM(
                    self.graph,
                    abs_weight_low=self.abs_weight_low,
                    abs_weight_high=self.abs_weight_high,
                    noise_scale=self.noise_scale,
                    contractive=contractive,
                    beta = self.beta
                )
            else:
                raise ValueError(f"Unknown sem_type: {self.sem_type}. Use one of ['lin', 'nnl', 'hyb']")

    def generate(self, interventions=True, change_targets=False, fixed_interventions=False, return_latents=False):
        dataset = list()
        if return_latents:
            latents = list()
        if interventions:

            if change_targets:
                self._pick_targets()

            for target_set in self.targets:
                generation_kwargs = dict(
                    n_samples=self.n_samples,
                    intervention_set=target_set,
                    fixed_intervention=fixed_interventions,
                    return_latents=return_latents,
                    intervention_scale=self.intervention_scale,
                    shift_scale=self.shift_scale,
                    noise_type=self.noise_type,
                )
                if self.sem_type == 'nnl':
                    generation_kwargs.update(
                        variable_shift_scale=self.variable_shift_scale,
                        shift_scale_range=self.shift_scale_range,
                        variable_intervention_scale=self.variable_intervention_scale,
                        intervention_scale_range=self.intervention_scale_range,
                    )
                data = self.gen_model.generateData(**generation_kwargs)
                if return_latents:
                    dataset.append(data[0])
                    latents.append(data[1])
                else:
                    dataset.append(data)
        
        else:
            data = self.gen_model.generateData(n_samples=self.n_samples, intervention_set=[None], return_latents=return_latents)
            if return_latents:
                dataset.append(data[0])
                latents.append(data[1])
            else:
                dataset.append(data)

        if return_latents:
            return dataset, latents

        return dataset 
    
    def get_adjacency(self):
        return nx.to_numpy_array(self.graph)

    def _pick_targets(self, max_iterations=100000):
        
        iter = 0
        if self.mode not in ['indiv-node', 'sat-pair-condition', 'no-constraint']:
            print(f"{self.mode} does not exist, defaulting to 'indiv-node'")
            self.mode = 'indiv-node'

        if self.mode == 'indiv-node':
            assert self.n_experiments == self.n_nodes, f"expected {self.n_nodes}, got {self.n_experiments}"
            self.targets = [np.array([node]) for node in range(self.n_nodes)]
            self.pair_condition = True

        else:
            not_correct = True
            self.targets = list()
            for _ in range(self.n_experiments):
                iter += 1
                n_targets = np.random.randint(self.min_targets, self.max_targets+1, 1)
                target_set = np.random.choice(self.n_nodes, n_targets, replace=False)
                self.targets.append(target_set)

                observed_set = np.setdiff1d(np.arange(self.n_nodes), target_set)
                indices = np.ix_(target_set, observed_set)
                self.pair_condition_matrix[indices] = 1.0

            if not self.checkPairCondition() and self.mode == 'sat-pair-condition':
                for node in range(self.n_nodes):
                    if self.pair_condition_matrix[node, :].sum() != self.n_nodes-1:
                        self.targets.append(np.array([node]))
                
                self.n_experiments = len(self.targets)

            self.pair_condition = self.checkPairCondition()

    def checkPairCondition(self):
        return self.pair_condition_matrix.sum() == self.n_nodes**2 - self.n_nodes

    def store_data(self, output_path, generate_data=False, datasets=None, fixed_intervention=False, interventions=True):
        if generate_data:
            datasets = self.generate(interventions=interventions, fixed_interventions=fixed_intervention)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, 'config.txt'), 'w') as file:
            file.write("n_nodes: {}\n".format(self.n_nodes))
            file.write("n_exps: {}\n".format(self.n_experiments))
            file.write("n_scale: {}\n".format(self.noise_scale))
            file.write("mode: {}\n".format(self.mode))
            file.write("sem_type: {}\n".format(self.sem_type))
            file.write("Interventions: {}\n".format(interventions))
        
        if interventions:
            for i, dataset in enumerate(datasets):
                np.save(os.path.join(output_path, 'dataset_{}.npy'.format(i)), dataset)
            
            np.save(os.path.join(output_path, 'intervention_sets.npy'), self.targets)

        else:
            np.save(os.path.join(output_path, 'dataset_0.npy'), datasets[0])
            intervention_sets = [[None]]
            np.save(os.path.join(output_path, 'intervention_sets.npy'), intervention_sets)
        
        if self.sem_type == 'lin':
            np.save(os.path.join(output_path, 'weights.npy'), self.gen_model.weights)
        if self.sem_type == 'nnl':
            adj_mat = get_adj_from_single_func(self.gen_model.f, self.gen_model.f.layers[0][0].weight.device)
            np.save(os.path.join(output_path, 'weights.npy'), adj_mat)
        if self.sem_type == 'hyb':
            np.save(os.path.join(output_path, 'weights.npy'), self.gen_model.weights)
        
        print("Stored data to path: {}".format(output_path))


def generate_data_learning(n_nodes_list, exp_dens_list, n_exp_list, n_samples, data_output_path, n_graphs=5, return_graph=True, sem_type='lin', n_hidden=0, act_fun='tanh', lip_const=0.99, mode='indiv-node', enforce_dag=False, contractive=True, interventions=True, variable_shift_scale=False, shift_scale_low=None, shift_scale_high=None, variable_intervention_scale=False, intervention_scale_low=None, intervention_scale_high=None):

    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)

    graphs_list = list()
    gen_model_list = list()

    for n_nodes, exp_dens, n_exp in zip(n_nodes_list, exp_dens_list, n_exp_list):
        graph_n_list = list()
        gen_model_n_list = list()
        for n in range(n_graphs):
            dataset_gen = Dataset(n_nodes=n_nodes,
                                  expected_density=exp_dens,
                                  n_samples=n_samples, 
                                  n_experiments=n_exp,
                                  mode=mode,
                                  sem_type=sem_type, 
                                  n_hidden=n_hidden, 
                                  act_fun=act_fun, 
                                  lip_constant=lip_const,
                                  enforce_dag=enforce_dag,
                                  contractive=contractive,
                                  variable_shift_scale=variable_shift_scale,
                                  shift_scale_low=shift_scale_low,
                                  shift_scale_high=shift_scale_high,
                                  variable_intervention_scale=variable_intervention_scale,
                                  intervention_scale_low=intervention_scale_low,
                                  intervention_scale_high=intervention_scale_high)
            graph_n_list.append(dataset_gen.graph)
            gen_model_n_list.append(dataset_gen.gen_model)
            datasets = dataset_gen.generate()
            data_path = os.path.join(data_output_path, 'nodes_{}/graph_{}'.format(n_nodes, n))
            dataset_gen.store_data(data_path, datasets, interventions=interventions)
        graphs_list.append(graph_n_list)
        gen_model_list.append(gen_model_n_list)

    if return_graph:
        return graphs_list, gen_model_list

def generate_data_validate(n_nodes_list, data_output_path, graphs_list, gen_model_list, n_targets_list, n_exp=10, n_samples=5000, sem_type='lin', variable_shift_scale=False, shift_scale_low=None, shift_scale_high=None, variable_intervention_scale=False, intervention_scale_low=None, intervention_scale_high=None):

    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
    
    n_graphs = len(graphs_list[0])

    for i, n_nodes in enumerate(n_nodes_list):
        for n in range(n_graphs):
            for n_targets in n_targets_list:
                dataset_gen = Dataset(n_nodes=n_nodes,
                                    expected_density=1,
                                    mode='no-constraint',
                                    graph_provided=True, 
                                    graph=graphs_list[i][n],
                                    gen_model_provided=True, 
                                    gen_model=gen_model_list[i][n],
                                    min_targets=n_targets, 
                                    max_targets=n_targets,
                                    n_samples=n_samples,
                                    n_experiments=n_exp,
                                    sem_type=sem_type,
                                    variable_shift_scale=variable_shift_scale,
                                    shift_scale_low=shift_scale_low,
                                    shift_scale_high=shift_scale_high,
                                    variable_intervention_scale=variable_intervention_scale,
                                    intervention_scale_low=intervention_scale_low,
                                    intervention_scale_high=intervention_scale_high)
                datasets = dataset_gen.generate(fixed_interventions=True)
                data_path = os.path.join(data_output_path, 'nodes_{}/graph_{}/n_inter_{}'.format(n_nodes, n, n_targets))
                dataset_gen.store_data(data_path, datasets)
    
# TODO add code to run from terminal
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--n-graphs', type=int, default=10)
    parser.add_argument('--dop', type=str, default='~/projects/datasets/linear_data')
    parser.add_argument('--sem-type', type=str, default='lin')
    parser.add_argument('--n-hidden', type=int, default=0)
    parser.add_argument('--act-fun', type=str, default='selu')
    parser.add_argument('--lip-const', type=float, default=0.99)
    parser.add_argument('--mode', type=str, default='indiv-node')
    parser.add_argument('--n-int-exp', type=int, default=10)
    parser.add_argument('--enf-dag', action='store_true', default=False)
    parser.add_argument('--n-contract', action='store_false', default=True)
    parser.add_argument('--no-inter', action='store_false', default=True)
    parser.add_argument('--variable-shift-scale', action='store_true', default=False)
    parser.add_argument('--shift-scale-low', type=float, default=None)
    parser.add_argument('--shift-scale-high', type=float, default=None)
    parser.add_argument('--variable-intervention-scale', action='store_true', default=False)
    parser.add_argument('--intervention-scale-low', type=float, default=None)
    parser.add_argument('--intervention-scale-high', type=float, default=None)

    args = parser.parse_args()

    n_nodes_list = [5, 10, 20]
    exp_dens_list = [1, 2, 2]
    n_exp_list = [5, 10, 20]
    train_path = os.path.join(args.dop, 'training_data')
    
    graph_list, gen_model_list = generate_data_learning(n_nodes_list,
                                                        exp_dens_list,
                                                        n_exp_list,
                                                        args.n_samples,
                                                        train_path,
                                                        args.n_graphs,
                                                        return_graph=True,
                                                        sem_type=args.sem_type,
                                                        n_hidden=args.n_hidden,
                                                        act_fun=args.act_fun,
                                                        lip_const=args.lip_const,
                                                        mode=args.mode,
                                                        enforce_dag=args.enf_dag,
                                                        contractive=args.n_contract,
                                                        interventions=args.no_inter,
                                                        variable_shift_scale=args.variable_shift_scale,
                                                        shift_scale_low=args.shift_scale_low,
                                                        shift_scale_high=args.shift_scale_high,
                                                        variable_intervention_scale=args.variable_intervention_scale,
                                                        intervention_scale_low=args.intervention_scale_low,
                                                        intervention_scale_high=args.intervention_scale_high)

    validation_path = os.path.join(args.dop, 'validation_data')
    n_targets_list = [2, 3, 4]
    generate_data_validate(n_nodes_list, 
                           data_output_path=validation_path,
                           graphs_list=graph_list,
                           gen_model_list=gen_model_list,
                           n_targets_list=n_targets_list,
                           n_exp=args.n_int_exp,
                           n_samples=args.n_samples,
                           sem_type=args.sem_type,
                           variable_shift_scale=args.variable_shift_scale,
                           shift_scale_low=args.shift_scale_low,
                           shift_scale_high=args.shift_scale_high,
                           variable_intervention_scale=args.variable_intervention_scale,
                           intervention_scale_low=args.intervention_scale_low,
                           intervention_scale_high=args.intervention_scale_high)

