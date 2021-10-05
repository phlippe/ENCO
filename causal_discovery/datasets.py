import torch
import torch.utils.data as data
import numpy as np
import time


class ObservationalCategoricalData(data.Dataset):

    def __init__(self, graph, dataset_size):
        """
        Dataset for simplifying the interaction with observational data
        in the distribution fitting stage. If the causal graph does not 
        have a pre-sampled dataset, a new dataset is sampled.

        Parameters
        ----------
        graph : CausalDAG
                The causal graph from which we want to get observational data from.
                If it has the attribute "data_obs", we use it as the dataset. 
                Otherwise, a new dataset is sampled.
        dataset_size : int
                       The size of the dataset to sample if no observational dataset
                       is provided in the first place. Otherwise, the minimum of
                       the exported dataset size and the requested dataset size
                       is used.
        """
        super().__init__()
        self.graph = graph
        self.var_names = [v.name for v in self.graph.variables]
        if not hasattr(self.graph, "data_obs"):
            start_time = time.time()
            print("Creating dataset...")
            data = graph.sample(batch_size=dataset_size, as_array=True)
            print("Dataset created in %4.2fs" % (time.time() - start_time))
        else:
            data = self.graph.data_obs
            if dataset_size <= data.shape[0]:
                data = data[:dataset_size]
            else:
                print('[WARNING - ObservationalCategoricalData] The requested dataset size is'
                      f' {dataset_size} but the exported graph\'s observational dataset has only'
                      f' {data.shape[0]} samples. Using {data.shape[0]} samples...')
        data = torch.from_numpy(data)
        self.data = correct_data_types(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class InterventionalDataset(object):

    def __init__(self, graph, dataset_size, batch_size, num_stacks=50):
        """
        Dataset for simplifying the interaction with interventional data
        in the graph fitting stage. If the causal graph does not have a
        pre-sampled dataset, a new dataset per variable is sampled. Since
        we have multiple variables to sample from, this dataset summarizes
        multiple data loaders and organizes the batch sampling via the
        'get_batch' method.

        Parameters
        ----------
        graph : CausalDAG
                The causal graph from which we want to get interventional data
                from. If it has the attribute "data_int", we use it as the dataset. 
                Otherwise, a new dataset is sampled.
        dataset_size : int
                       Number of samples per variable to sample if no
                       interventional dataset is provided in the first place.
                       Otherwise, the minimum of the exported dataset size 
                       and the requested dataset size is used.
        batch_size : int
                     Number of samples in a batch that is returned via the 
                     'get_batch' function.
        num_stacks : int
                     This parameter is only used if no interventional dataset
                     is provided. It determines how many variables to sample from
                     simultaneously for faster processing speed. It has no effect
                     on the actual dataset afterwards.
        """
        self.graph = graph
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.data_loaders = {}
        self.data_iter = {}

        if hasattr(self.graph, "data_int"):
            if self.graph.data_int.shape[1] < self.dataset_size:
                print('[WARNING - InterventionalDataset] The requested dataset size is'
                      f' {self.dataset_size} but the exported graph\'s interventional'
                      f' dataset has only {self.graph.data_int.shape[1]} samples.'
                      f' Using {self.graph.data_int.shape[1]} samples...')
            for var_idx in range(self.graph.num_vars):
                self._add_dataset(self.graph.data_int[var_idx], var_idx)
        else:
            print("Sampling interventional data...")
            start_time = time.time()
            intervention_list = []
            for var_idx in range(self.graph.num_vars):
                var = self.graph.variables[var_idx]
                values = np.random.randint(var.prob_dist.num_categs,
                                           size=(dataset_size,))  # Uniform interventional distribution
                intervention_list.append((var_idx, var, values))
                if len(intervention_list) >= num_stacks:
                    self._add_vars(intervention_list)
                    intervention_list = []
            if len(intervention_list) > 0:
                self._add_vars(intervention_list)
            print("Done in %4.2fs" % (time.time() - start_time))

    def _add_vars(self, intervention_list):
        """
        Helper function for sampling interventional dataset
        """
        num_vars = len(intervention_list)
        intervention_dict = {}
        for i, (var_idx, var, values) in enumerate(intervention_list):
            v_array = -np.ones((num_vars, self.dataset_size), dtype=np.int32)
            v_array[i] = values
            v_array = np.reshape(v_array, (-1,))
            intervention_dict[var.name] = v_array
        int_sample = self.graph.sample(interventions=intervention_dict,
                                       batch_size=self.dataset_size*num_vars,
                                       as_array=True)
        int_sample = torch.from_numpy(int_sample).reshape(num_vars, self.dataset_size, int_sample.shape[-1])
        for i, (var_idx, var, values) in enumerate(intervention_list):
            self._add_dataset(int_sample[i], var_idx)

    def _add_dataset(self, samples, var_idx):
        """
        Helper function for sampling interventional dataset
        """
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        samples = correct_data_types(samples)
        if self.dataset_size <= samples.shape[0]:
            samples = samples[:self.dataset_size]
        dataset = data.TensorDataset(samples)
        self.data_loaders[var_idx] = data.DataLoader(dataset, batch_size=self.batch_size,
                                                     shuffle=True, pin_memory=False, 
                                                     drop_last=(len(dataset)>self.batch_size))
        self.data_iter[var_idx] = iter(self.data_loaders[var_idx])

    def get_batch(self, var_idx):
        """
        Returns batch of interventional data for specified variable.
        """
        try:
            batch = next(self.data_iter[var_idx])
        except StopIteration:
            self.data_iter[var_idx] = iter(self.data_loaders[var_idx])
            batch = next(self.data_iter[var_idx])
        return batch[0]


def correct_data_types(data):
    if data.dtype in [torch.uint8, torch.int16, torch.int32]:
        data = data.long()
    elif data.dtype in [torch.float16, torch.float64]:
        data = data.float()
    return data
