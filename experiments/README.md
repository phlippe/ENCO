# Experiments

Experiments can be run two modi, denoted by different files.
* `run_exported_graphs.py` applies ENCO to a causal graph that has been saved on the disk. The graph can be saved in the following three formats:
   * `.bif` format as from the BnLearn repository
   * `.npz` format as generated when running [`causal_graphs/graph_export.py`](../causal_graphs/graph_export.py)
   * `.pt` format as when saving a `CausalDAG` object to disk ([`CausalDAG.save_to_file`](../causal_graphs/graph_definition.py))
  The graph files can be specified using the parser argument `--graph_files`. Multiple files can be specified when ENCO should be tested on all those graphs in sequence. Example usage:
  ```bash
  python run_exported_graphs.py --graph_files ../causal_graphs/real_data/small_graphs/sachs.bif
  ```
* `run_generated_graphs.py` applies ENCO to newly generated causal graphs. It takes additional arguments for the graph(s) to generate, and is mostly meant for prototyping on various graph structures such as in the sythetic dataset. Example usage:
  ```bash
  python run_generated_graphs.py --graph_type random --num_vars 25 --edge_prob 0.3 --num_graphs 2
  ```

For all experiments, checkpoint folders are created that store logging information and the final, predicted graph. By default, those are created under the folder `checkpoints` with a date and time folder string. To specify a different checkpoint directory, use the argument `--checkpoint_dir`.

## Experiments from the paper
The commands to reproduce the experiments in the paper are summarized in the folder [run_scripts](run_scripts).
See the corresponding README for details.
