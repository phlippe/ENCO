import torch
from tqdm.auto import tqdm
import matplotlib
import itertools
import numpy as np

# Set constant below to True if no GPU should be used. Otherwise, GPU will be used by default if exists.
CPU_ONLY = False
# If CLUSTER is False, progress bars are shown via tqdm. Otherwise, they are surpressed to reduce stdout.
CLUSTER = False

def get_device():
    return torch.device("cuda:0" if (not CPU_ONLY and torch.cuda.is_available()) else "cpu")

def set_cluster(is_cluster):
    global CLUSTER 
    CLUSTER = is_cluster
    if CLUSTER:
        matplotlib.use('Agg')

def is_cluster():
    return CLUSTER

def track(iterator, **kwargs):
    if not CLUSTER:
        return tqdm(iterator, **kwargs)
    else:
        return iterator

############################
## FINDING ACYCLIC GRAPHS ##
############################

@torch.no_grad()
def find_best_acyclic_graph(pred_matrix=None, gamma=None, theta=None):
    """
    Given the set of parameters theta and gamma, find the most likeliest acyclic graph
    by finding the order of variables that maximizes the orientation probabilities of theta.
    We use a simplified heuristic implementation which showed to work well for the normal
    usecases, but can be further optimized in accuracy or efficiency.
    """
    if gamma is None or theta is None:
        assert pred_matrix is not None, 'The input pred_matrix must be not None if gamma or theta are not provided.'
        gamma, theta = pred_matrix.clone().unbind(dim=0)
    gamma, theta = gamma.cpu(), theta.cpu()
    theta = theta.float()
    hard_matrix = ((gamma > 0.5) * (theta > 0.5))
    # Find all cycles in the current graph
    cycle_frames = find_cycles(hard_matrix)
    # For each cycle as list of nodes, we find the order of those that maximizes 
    # the product of orientation probabilities.
    for frame in cycle_frames:
        if len(frame) < 7:
            # Brute-Force: test all permutations
            permutations = itertools.permutations(range(len(frame)))
        else:
            # Greedy: what is the best move of a single variable
            default_permut = list(range(len(frame)))
            permutations = [default_permut]
            for i in range(len(default_permut)):
                e = default_permut[i]
                rest = default_permut[:i] + default_permut[i+1:]
                permutations += [[rest[:j]+[e]+rest[j:]] for j in range(0,len(rest)+1) if j != i]
        # For numerical stability, we add the log probabilities instead of
        # multiplying the raw probabilities 
        small_theta = (theta[frame][:,frame]+1e-10).log()
        if torch.isnan(small_theta).any():
            print('Found some NaNs...', small_theta)
        best_score, best_permut = -float('inf'), None
        # Find permutation with highest log probability
        for permut in permutations:
            permut = list(permut)
            perm_theta = small_theta[permut][:,permut]
            score = torch.triu(perm_theta, diagonal=1).sum()
            if score > best_score:
                best_score = score
                best_permut = permut
        # Apply best permutation
        triu = torch.triu(torch.ones(len(frame), len(frame)), diagonal=1)
        rev_permut = [best_permut.index(i) for i in range(len(frame))]
        triu = triu[rev_permut][:,rev_permut]
        for i,f in enumerate(frame):
            theta[f,frame] = triu[i]
            
    hard_matrix = ((gamma > 0.5) * (theta > 0.5))
    return hard_matrix

def find_cycles(adj_matrix):
    """
    Given an adjacency matrix, return all cycles in the graph.
    Cycles are returned as list of nodes in the cycle, and might
    not be unique.
    """
    # For efficiency, we find possible cycles by checking for edges from X_i->X_j where i>j
    # since all ground truth graphs are sorted from first to last variable in the causal 
    # ordering. Note that this does *not* influence which cycles we find, just more efficient.
    rev_edges = []
    for i in range(adj_matrix.shape[0]):
        for j in range(i,adj_matrix.shape[1]):
            if adj_matrix[j,i]:
                rev_edges.append((i,j))

    # Find all nodes that build up the cycles from node X_i to X_j
    rev_edges = [(i,max([j for k,j in rev_edges if k==i])) for i, _ in rev_edges]
    cycle_frames = []
    for i,j in rev_edges:
        nodes = find_nodes_on_paths(adj_matrix,i,j)
        if nodes is None:
            continue
        nodes = torch.where(nodes == 1)[0].numpy().tolist()
        frame = nodes
        if len(frame) == 0:
            continue
        cycle_frames.append(frame)

    # Remove duplicate cycles
    cycle_frames = [sorted(f) for f in cycle_frames]
    list_2 = cycle_frames[:]
    b = 0
    for i, f in enumerate(list_2):
        if f in cycle_frames[:(i-b)]:
            del cycle_frames[i-b]
            b += 1

    return cycle_frames

def find_nodes_on_paths(adj_matrix, source_node, target_node, nodes_on_path=None, current_path=None):
    """
    Find all nodes that are parts of paths from the source node to the target node.
    Simple, recursive algorithm: iterate for all children of the source node. 
    """
    # We store a binary 'visiting' mask in nodes_on_path. In every iteration,
    # we set nodes_on_path[source_node] to 1 if we can reach the target node
    # from it.
    if nodes_on_path is None:
        nodes_on_path = torch.zeros(adj_matrix.shape[0])
    if current_path is None:
        current_path = torch.zeros(adj_matrix.shape[0])
    current_path[source_node] = 1
    
    if source_node == target_node:  # Found target node
        nodes_on_path[source_node] = 1
        return nodes_on_path
    elif nodes_on_path[source_node] == 1:  # We already visited this node and reached the target node
        return nodes_on_path
    elif nodes_on_path[source_node] == -1:  # We already visited this node and cannot reach the target node
        return None
    else:
        # Start search for every child of the source node
        children = torch.where(adj_matrix[source_node])[0]
        for c in children:
            if current_path[c] == 1:
                continue
            ret = find_nodes_on_paths(adj_matrix, c, target_node, nodes_on_path=nodes_on_path, current_path=np.copy(current_path))
            if ret is not None:  # If True, we have reached the target node from the child
                nodes_on_path[source_node] = 1

        if nodes_on_path[source_node] <= 0:  # If True, we have not reached the target node from any child
            nodes_on_path[source_node] = -1
            return None 
        else:
            return nodes_on_path