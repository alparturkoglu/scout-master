from torch.utils.data import Dataset 
import numpy as np

class experimentDataset(Dataset):
    def __init__(self, dataset, intervention_set):
        self.dataset = dataset
        self.intervention_set = intervention_set 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class experimentDatasetStrat(Dataset):
    def __init__(self, datasets, intervention_sets):
        self.intervention_sets = intervention_sets 
        self.datasets = datasets

        self.make_final_data()

    def make_final_data(self):
        self.final_data = np.vstack(self.datasets)
        masks = list()
        for dataset, intervention_set in zip(self.datasets, self.intervention_sets):
            mask = np.ones(dataset.shape)
            if intervention_set[0] != None:
                mask[:, intervention_set] = 0
            
            masks.append(mask)
        
        self.masks = np.vstack(masks)

    def __len__(self):
        return len(self.final_data)

    def __getitem__(self, idx):
        return self.final_data[idx], self.masks[idx]
    
# --- Option A: Mixed Experiment DataLoader (randomly mixes samples from all experiments) ---
class MixedExperimentDataset(Dataset):
    def __init__(self, datasets, intervention_sets):
        # datasets: list of np arrays (n_samples_i x n_nodes)
        # intervention_sets: list of lists/arrays of intervened node indices for each dataset
        self.datasets = datasets
        self.intervention_sets = intervention_sets
        
        # Verify that datasets and intervention_sets have the same length
        if len(datasets) != len(intervention_sets):
            raise ValueError(f"Number of datasets ({len(datasets)}) must match number of intervention_sets ({len(intervention_sets)})")
        
        self.data = np.vstack(datasets)  # stacked samples
        # create per-sample experiment id
        self.exp_ids = np.concatenate([np.full(len(d), i, dtype=np.int64) for i, d in enumerate(datasets)])
        # build per-sample masks: 1 if observed, 0 if intervened
        masks = []
        for d, inter in zip(datasets, intervention_sets):
            # inter may be [None] meaning no intervention; treat None as all observed
            if inter is None or (isinstance(inter, (list, tuple)) and len(inter) == 0) or (isinstance(inter, list) and inter[0] is None):
                m = np.ones_like(d)
            else:
                m = np.ones_like(d)
                m[:, inter] = 0
            masks.append(m)
        self.masks = np.vstack(masks).astype(np.float32)
        
        # Final sanity check
        if len(self.data) != len(self.masks):
            raise ValueError(f"Data length ({len(self.data)}) must match masks length ({len(self.masks)})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        mask = self.masks[idx]  # shape: (n_nodes,)
        exp_id = int(self.exp_ids[idx])
        return x, mask, exp_id

