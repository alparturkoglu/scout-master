import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np


DEFAULT_SERGIO_ROOT = Path(__file__).resolve().parents[2] / "SERGIO-master"


def _coerce_rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _load_sergio_class(sergio_root=None):
    sergio_root = Path(sergio_root) if sergio_root is not None else DEFAULT_SERGIO_ROOT
    if str(sergio_root) not in sys.path:
        sys.path.insert(0, str(sergio_root))

    from SERGIO.sergio import sergio

    return sergio


def sample_random_cyclic_sergio_graph(
    n_nodes,
    edge_prob=0.25,
    n_master_regulators=None,
    min_cycle_len=3,
    seed=None,
):
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least 3 to guarantee a directed cycle.")

    rng = _coerce_rng(seed)
    if n_master_regulators is None:
        n_master_regulators = max(1, n_nodes // 5)
    n_master_regulators = int(np.clip(n_master_regulators, 1, n_nodes - 2))

    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    master_regulators = np.arange(n_master_regulators, dtype=int)
    target_genes = np.arange(n_master_regulators, n_nodes, dtype=int)

    for dst in target_genes:
        for src in range(n_nodes):
            if src != dst and rng.random() < edge_prob:
                adjacency[src, dst] = 1.0

    cycle_len = int(np.clip(min_cycle_len, 2, len(target_genes)))
    cycle_nodes = rng.choice(target_genes, size=cycle_len, replace=False)
    for src, dst in zip(cycle_nodes, np.roll(cycle_nodes, -1)):
        adjacency[int(src), int(dst)] = 1.0

    for src in master_regulators:
        if adjacency[src].sum() == 0:
            dst = int(rng.choice(target_genes))
            adjacency[src, dst] = 1.0

    for dst in target_genes:
        if adjacency[:, dst].sum() == 0:
            src_candidates = np.delete(np.arange(n_nodes, dtype=int), np.where(np.arange(n_nodes) == dst))
            src = int(rng.choice(src_candidates))
            adjacency[src, dst] = 1.0

    np.fill_diagonal(adjacency, 0.0)
    return adjacency, master_regulators


def sample_sergio_interaction_matrix(adjacency, weight_low=1.0, weight_high=5.0, seed=None):
    rng = _coerce_rng(seed)
    adjacency = np.asarray(adjacency, dtype=float)
    edge_weights = rng.uniform(weight_low, weight_high, size=adjacency.shape)
    edge_signs = rng.choice([-1.0, 1.0], size=adjacency.shape)
    interaction_matrix = edge_weights * edge_signs * adjacency
    return interaction_matrix, edge_weights, edge_signs


def infer_master_regulators(adjacency):
    adjacency = np.asarray(adjacency, dtype=float)
    return np.flatnonzero(adjacency.sum(axis=0) == 0).astype(int)


def _write_sergio_inputs(workdir, adjacency, interaction_matrix, master_regulators, mr_rates, hill):
    interaction_path = Path(workdir) / "interaction.txt"
    regs_path = Path(workdir) / "regs.txt"
    master_regulator_set = set(int(gene) for gene in np.asarray(master_regulators, dtype=int))

    with interaction_path.open("w") as handle:
        for target in range(adjacency.shape[0]):
            if target in master_regulator_set:
                continue

            regulators = np.flatnonzero(adjacency[:, target]).astype(int)
            if regulators.size == 0:
                raise ValueError(
                    f"Gene {target} has no incoming edges, so it must be listed as a master regulator."
                )

            row = [str(target), str(len(regulators))]
            row.extend(str(int(reg)) for reg in regulators)
            row.extend(f"{float(interaction_matrix[reg, target]):.6f}" for reg in regulators)
            row.extend([str(float(hill))] * len(regulators))
            handle.write(",".join(row) + "\n")

    with regs_path.open("w") as handle:
        for idx, gene in enumerate(np.asarray(master_regulators, dtype=int)):
            row = [str(int(gene))]
            row.extend(f"{float(rate):.6f}" for rate in mr_rates[idx])
            handle.write(",".join(row) + "\n")

    return interaction_path, regs_path


def _expr_to_cell_gene_matrix(expr):
    expr = np.concatenate(expr, axis=1)
    return np.asarray(expr.T, dtype=float)


def _simulate_environment(
    sergio_cls,
    interaction_path,
    regs_path,
    n_nodes,
    num_cell_types,
    num_cells_per_type,
    noise_params,
    decay,
    sampling_state,
    noise_type,
    dt,
    hill,
    knockout_gene=None,
):
    sim = sergio_cls(
        number_genes=n_nodes,
        number_bins=num_cell_types,
        number_sc=num_cells_per_type,
        noise_params=noise_params,
        decays=decay,
        sampling_state=sampling_state,
        noise_type=noise_type,
        dt=dt,
    )
    sim.build_graph(
        input_file_taregts=str(interaction_path),
        input_file_regs=str(regs_path),
        shared_coop_state=hill,
    )
    sim.simulate(knockout_gene=knockout_gene)
    return _expr_to_cell_gene_matrix(sim.getExpressions())


def generate_sergio_dataset_from_graph(
    adjacency,
    sergio_root=None,
    master_regulators=None,
    interaction_matrix=None,
    n_intervention_sets=None,
    num_cell_types=3,
    num_cells_per_type=40,
    noise_params=1.0,
    decay=0.8,
    sampling_state=5,
    noise_type="dpd",
    dt=0.01,
    hill=2,
    weight_low=1.0,
    weight_high=5.0,
    master_rate_low=0.5,
    master_rate_high=3.0,
    standardize=True,
    seed=None,
):
    rng = _coerce_rng(seed)
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix.")

    adjacency = adjacency.copy()
    np.fill_diagonal(adjacency, 0.0)
    n_nodes = adjacency.shape[0]

    if master_regulators is None:
        master_regulators = infer_master_regulators(adjacency)
    master_regulators = np.asarray(master_regulators, dtype=int)

    if interaction_matrix is None:
        interaction_matrix, edge_weights, edge_signs = sample_sergio_interaction_matrix(
            adjacency,
            weight_low=weight_low,
            weight_high=weight_high,
            seed=rng,
        )
    else:
        interaction_matrix = np.asarray(interaction_matrix, dtype=float)
        edge_weights = np.abs(interaction_matrix)
        edge_signs = np.sign(interaction_matrix)

    if master_regulators.size > 0:
        mr_rates = rng.uniform(
            master_rate_low,
            master_rate_high,
            size=(master_regulators.size, num_cell_types),
        )
    else:
        mr_rates = np.zeros((0, num_cell_types), dtype=float)

    if n_intervention_sets is None:
        n_intervention_sets = min(n_nodes, 5)
    n_intervention_sets = int(min(n_intervention_sets, n_nodes))
    ko_genes = rng.choice(np.arange(n_nodes, dtype=int), size=n_intervention_sets, replace=False)

    sergio_cls = _load_sergio_class(sergio_root)
    with tempfile.TemporaryDirectory(prefix="sergio_cyclic_") as tmpdir:
        interaction_path, regs_path = _write_sergio_inputs(
            tmpdir,
            adjacency=adjacency,
            interaction_matrix=interaction_matrix,
            master_regulators=master_regulators,
            mr_rates=mr_rates,
            hill=hill,
        )

        raw_training_data = [
            _simulate_environment(
                sergio_cls,
                interaction_path=interaction_path,
                regs_path=regs_path,
                n_nodes=n_nodes,
                num_cell_types=num_cell_types,
                num_cells_per_type=num_cells_per_type,
                noise_params=noise_params,
                decay=decay,
                sampling_state=sampling_state,
                noise_type=noise_type,
                dt=dt,
                hill=hill,
                knockout_gene=None,
            )
        ]
        intervention_sets = [[np.array([], dtype=int)]]

        for gene in ko_genes:
            raw_training_data.append(
                _simulate_environment(
                    sergio_cls,
                    interaction_path=interaction_path,
                    regs_path=regs_path,
                    n_nodes=n_nodes,
                    num_cell_types=num_cell_types,
                    num_cells_per_type=num_cells_per_type,
                    noise_params=noise_params,
                    decay=decay,
                    sampling_state=sampling_state,
                    noise_type=noise_type,
                    dt=dt,
                    hill=hill,
                    knockout_gene=int(gene),
                )
            )
            intervention_sets.append([np.array([int(gene)], dtype=int)])

    if standardize:
        all_data = np.concatenate(raw_training_data, axis=0)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        std = np.where(std == 0.0, 1.0, std)
        training_data = [(dataset - mean) / std for dataset in raw_training_data]
    else:
        mean = None
        std = None
        training_data = raw_training_data

    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    return {
        "training_data": training_data,
        "raw_training_data": raw_training_data,
        "intervention_sets": intervention_sets,
        "n_experiments": len(training_data),
        "gt_graph": adjacency,
        "graph": graph,
        "interaction_matrix": interaction_matrix,
        "edge_weights": edge_weights,
        "edge_signs": edge_signs,
        "ko_genes": ko_genes,
        "master_regulators": master_regulators,
        "mr_rates": mr_rates,
        "mean": mean,
        "std": std,
    }


def generate_random_cyclic_sergio_dataset(
    n_nodes,
    sergio_root=None,
    edge_prob=0.25,
    n_master_regulators=None,
    min_cycle_len=3,
    n_intervention_sets=None,
    num_cell_types=3,
    num_cells_per_type=40,
    noise_params=1.0,
    decay=0.8,
    sampling_state=5,
    noise_type="dpd",
    dt=0.01,
    hill=2,
    weight_low=1.0,
    weight_high=5.0,
    master_rate_low=0.5,
    master_rate_high=3.0,
    standardize=True,
    seed=None,
):
    rng = _coerce_rng(seed)
    adjacency, master_regulators = sample_random_cyclic_sergio_graph(
        n_nodes=n_nodes,
        edge_prob=edge_prob,
        n_master_regulators=n_master_regulators,
        min_cycle_len=min_cycle_len,
        seed=rng,
    )
    return generate_sergio_dataset_from_graph(
        adjacency=adjacency,
        sergio_root=sergio_root,
        master_regulators=master_regulators,
        n_intervention_sets=n_intervention_sets,
        num_cell_types=num_cell_types,
        num_cells_per_type=num_cells_per_type,
        noise_params=noise_params,
        decay=decay,
        sampling_state=sampling_state,
        noise_type=noise_type,
        dt=dt,
        hill=hill,
        weight_low=weight_low,
        weight_high=weight_high,
        master_rate_low=master_rate_low,
        master_rate_high=master_rate_high,
        standardize=standardize,
        seed=rng,
    )
