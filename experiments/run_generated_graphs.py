import json
import os
from datetime import datetime
import sys
sys.path.append("../")

from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func
from causal_discovery.utils import set_cluster
from experiments.utils import set_seed, get_basic_parser, test_graph


if __name__ == '__main__':
    parser = get_basic_parser()
    parser.add_argument('--graph_type', type=str, default='random',
                        help='Which graph type to test on. Currently supported are: '
                             'chain, bidiag, collider, jungle, full, regular, random, '
                             'random_max_#N where #N is to be replaced with an integer. '
                             'random_max_10 is random with max. 10 parents per node.')
    parser.add_argument('--num_graphs', type=int, default=1,
                        help='Number of graphs to generate and sequentially test on.')
    parser.add_argument('--num_vars', type=int, default=8,
                        help='Number of variables that the graphs should have.')
    parser.add_argument('--num_categs', type=int, default=10,
                        help='Number of categories/different values each variable can take.')
    parser.add_argument('--edge_prob', type=float, default=0.2,
                        help='For random graphs, the probability of two arbitrary nodes to be connected.')
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

    for gindex in range(args.num_graphs):
        # Seed setting for reproducibility
        set_seed(args.seed+gindex)  # Need to increase seed, otherwise we might same graphs
        # Generate graph
        print("Generating %s graph with %i variables..." % (args.graph_type, args.num_vars))
        graph = generate_categorical_graph(num_vars=args.num_vars,
                                           min_categs=args.num_categs,
                                           max_categs=args.num_categs,
                                           edge_prob=args.edge_prob,
                                           connected=True,
                                           use_nn=True,
                                           graph_func=get_graph_func(args.graph_type),
                                           seed=args.seed+gindex)
        file_id = "%s_%s" % (str(gindex+1).zfill(3), args.graph_type)
        # Save graph
        graph.save_to_file(os.path.join(checkpoint_dir, "graph_%s.pt" % (file_id)))
        # Visualize graph
        if graph.num_vars <= 100:
            print("Visualizing graph...")
            figsize = max(3, graph.num_vars ** 0.7)
            visualize_graph(graph,
                            filename=os.path.join(checkpoint_dir, "graph_%s.pdf" % (file_id)),
                            figsize=(figsize, figsize),
                            layout="circular" if graph.num_vars < 40 else "graphviz")

        # Start structure learning
        test_graph(graph, args, checkpoint_dir, file_id)
