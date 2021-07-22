"""
File with functions for visualizing graphs.
"""
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt


def visualize_graph(graph, **kwargs):
    """
    Visualizes a CausalDAG object.

    Parameters
    ----------
    graph : CausalDAG
            The graph to visualize
    **kwargs : dict
               Additional arguments to pass to 'graph_to_image' or plt.figure
    """
    G = nx.DiGraph()
    G.add_nodes_from([v.name for v in graph.variables])
    edges = [[graph.variables[v_idx].name for v_idx in e] for e in graph.edges.tolist()]
    G.add_edges_from(edges)
    graph_to_image(G, **kwargs)


def graph_to_image(G, filename=None, show_plot=False, layout="graphviz", **kwargs):
    """
    Takes a networkx-graph object, and plots it with matplotlib.

    Parameters
    ----------
    G : nx.DiGraph
        Graph represented in networkx.
    filename : str
               The path to save this figure to. If None, the figure will not be saved.
    show_plot : bool
                If True, it runs 'plt.show' at the end of the function.
    layout : str
             Layout to use for visualizing the graph. Options are 'graphviz', 'circular',
             and 'planar'
    **kwargs : dict
               Additional arguments to pass to plt.figure, e.g. the figure size.
    """
    _ = plt.figure(**kwargs)
    if layout == "graphviz":
        pos = graphviz_layout(G, prog="dot")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "planar":
        pos = nx.planar_layout(G)
    else:
        pos = None
    nx.draw(G, pos,
            arrows=True,
            with_labels=True,
            font_weight='bold',
            node_color='lightgrey',
            edgecolors='black',
            node_size=600,
            arrowstyle='-|>',
            arrowsize=16)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', transparent=True)
    if show_plot:
        plt.show()
    plt.close()
