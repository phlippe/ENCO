"""
Utilities for loading a graph in 'bif' format from the BnLearn repository.
"""
import re
import numpy as np
import sys
sys.path.append("../")

from causal_graphs.variable_distributions import CategoricalDist, CategProduct
from causal_graphs.graph_definition import CausalVariable, CausalDAG


def load_graph_file(filename):
    """
    Loads a graph specified in .bif format into a CausalDAG object. Used to process real-world datasets.

    Parameters
    ----------
    filename : str
               Path to the file that should be loaded as graph.
    """
    with open(filename, "r") as f:
        text = f.read()

    variables_str = re.findall("variable .*\{\n?.*\n?\}", text)
    variables = dict()
    for v_str in variables_str:
        v = dict()
        name = v_str.split("variable ")[1].split("{")[0].strip()
        v["categs"] = [s.strip() for s in v_str.split("{")[-1].split("}")[0].split(",")]
        v["num_categs"] = len(v["categs"])
        variables[name] = v

    prob_str = re.findall("probability .*\{\n[^}]*\}", text)
    probabilities = dict()
    edges = []
    for p_str in prob_str:
        bracks = p_str.split("probability (")[1].split(")")[0]
        if "|" in bracks:
            out = bracks.split("|")[0].strip()
            inputs = [s.strip() for s in bracks.split("|")[1].split(",")]
            array_shape = [variables[inp_name]["num_categs"] for inp_name in inputs] + [variables[out]["num_categs"]]
            prob_array = np.zeros(tuple(array_shape), dtype=np.float32)

            for l in p_str.split("\n")[1:-1]:
                inp_vals = re.findall("\(.*\)", l)[0][1:-1].split(",")
                inp_indices = [variables[name]["categs"].index(val.strip()) for name, val in zip(inputs, inp_vals)]
                inp_indices = tuple(inp_indices)
                val_str = re.findall("\).*;", l)[0][1:-1]
                prob_array[inp_indices] = np.array([float(v) for v in val_str.split(",")])
        else:
            out, inputs = bracks.strip(), []
            val_str = re.findall("table .*;", p_str)[0][6:-1]
            prob_array = np.array([float(v) for v in val_str.split(",")])

        probabilities[out] = CategProduct(input_names=inputs, val_grid=prob_array)
        edges += [[inp, out] for inp in inputs]

    variables = [CausalVariable(name, CategoricalDist(variables[name]["num_categs"],
                                                      probabilities[name])) for name in variables]
    var_names = [v.name for v in variables]
    edges = [[var_names.index(v) for v in e] for e in edges]
    graph = CausalDAG(variables, edges=edges)
    return graph


if __name__ == '__main__':
    """
    Run this file to get a visualization of all graphs in the real_data folder.
    """
    from glob import glob
    from causal_graphs.graph_visualization import visualize_graph

    files = sorted(glob("real_data/*.bif"))
    for f in files:
        graph = load_graph_file(f)
        print(f, "-> %i nodes, %i categories overall" %
              (graph.num_vars, sum([v.prob_dist.num_categs for v in graph.variables])))
        figsize = max(3, graph.num_vars ** 0.5)
        visualize_graph(graph, filename=f.split(".bif")[0]+"_graph.pdf", figsize=(figsize, figsize))
