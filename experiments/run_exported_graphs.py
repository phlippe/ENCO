import json
import os
from datetime import datetime
import sys
sys.path.append("../")

from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.graph_export import load_graph
from causal_graphs.graph_real_world import load_graph_file
from causal_graphs.graph_definition import CausalDAG
from causal_discovery.utils import set_cluster
from experiments.utils import set_seed, get_basic_parser, test_graph


if __name__ == '__main__':
    parser = get_basic_parser()
    parser.add_argument('--graph_files', type=str, nargs='+',
                        help='Graph files to apply ENCO to. Files must be .pt, .npz, or .bif files.')
    args = parser.parse_args()

    # Basic checkpoint directory creation
    current_date = datetime.now()
    if args.checkpoint_dir is None or len(args.checkpoint_dir) == 0:
        checkpoint_dir = "checkpoints/%02d_%02d_%02d__%02d_%02d_%02d/" % (
            current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, current_date.second)
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    set_cluster(args.cluster)

    for gindex, graph_path in enumerate(args.graph_files):
        # Seed setting for reproducibility
        set_seed(args.seed)
        # Load graph
        if graph_path.endswith(".bif"):
            graph = load_graph_file(graph_path)
        elif graph_path.endswith(".pt"):
            graph = CausalDAG.load_from_file(graph_path)
        elif graph_path.endswith(".npz"):
            graph = load_graph(graph_path)
        else:
            assert False, "Unknown file extension for " + graph_path
        graph_name = graph_path.split("/")[-1].rsplit(".", 1)[0]
        if graph_name.startswith("graph_"):
            graph_name = graph_name.split("graph_")[-1]
        file_id = "%s_%s" % (str(gindex+1).zfill(3), graph_name)
        # Visualize graph
        if graph.num_vars <= 100:
            figsize = max(3, graph.num_vars ** 0.7)
            visualize_graph(graph,
                            filename=os.path.join(checkpoint_dir, "graph_%s.pdf" % (file_id)),
                            figsize=(figsize, figsize),
                            layout="circular" if graph.num_vars < 40 else "graphviz")
        s = "== Testing graph \"%s\" ==" % graph_name
        print("="*len(s)+"\n"+s+"\n"+"="*len(s))
        # Start structure learning
        test_graph(graph, args, checkpoint_dir, file_id)
