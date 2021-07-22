import torch
from tqdm.auto import tqdm
import matplotlib

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

