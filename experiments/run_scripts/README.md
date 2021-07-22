## Bash scripts
The scripts in this folder contain the commands for running the experiments presented in the paper. Note:
* Download the datasets via `download_datasets.sh` before running any experiment except `experiment_scale.sh`.
* Run the bash scripts from this directory, since they jump back to the `experiments` folder.
* The experiments have been designed for a single GPU device with 24GB memory. The full memory is only necessary for experiments with large graphs (>100 variables). If you have a smaller GPU and run out of memory during the graph fitting stage, adjust the argument `max_graph_stacking` to a smaller value. The used memory scales approximately linearly with `max_graph_stacking`. In case you run out of memory during the distribution fitting stage, adjust the argument `batch_size` and increase `GF_num_batches` to keep the overall batch size for the graph fitting stage constant.
* The experiments run by default in `cluster` mode which reduces the output. If you remove the `cluster` argument in the bash scripts, the training progress will be shown with progress bars.
* All checkpoints and logging information can be found in `experiments/checkpoints/...` by default.