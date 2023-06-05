"""
This script handles the calculation of various network metrics on the given graph.
Currently used for the Twitter and Enron networks, however, any graph in gpickle
format can be given as an argument.

For usage, run this script from the terminal with the --help command
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib
import scipy.stats
import scipy.special
import scipy.interpolate
import math

ENRON_PATH = '.\\Networks\\enron_network.csv' # Default Enron network path
TWITTER_PATH = '.\\Networks\\twitter_network_names.csv' # Default Twitter network path

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default=ENRON_PATH, help='Source path to graph (in Gpickle format)')
parser.add_argument('--out', type=str, default='enron_results.json', help='Output path')
parser.add_argument('--n_reps', type=int, default=5, help='Number of repetitions for betweenness calculation')
parser.add_argument('--node_frac', type=float, default=0.5, help='Fraction of nodes to sample from the graph for betweenness calculation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
    

def plot_dist(graph : nx.Graph | nx.DiGraph | None = None, degrees : list | np.array | None = None, deg_type='in'):
    """
    Plots the degree distribution of a given graph using logarithimc
    binning. If the degrees is not None, uses this array as the 
    graph degrees

    :param graph: A nx.Graph or DiGraph instance. If not provided, degrees cannot be None
    :param degrees: List of node degrees in the graph
    :param deg_type: Type of degree to consider. Available options: 'in', 'out'
    """
    fig, axes = plt.subplots(2, 2)

    if degrees is None:
        if deg_type == 'in':
            degrees = np.array([d for n, d in graph.in_degree() if d != 0]) # Ignore 0 degree nodes

        if deg_type == 'out':
            degrees = np.array([d for n, d in graph.out_degree() if d != 0]) # Ignore 0 degree nodes

    degree_vals, degree_counts = np.unique(degrees, return_counts=True)

    # Use logarithmic binning to create a histogram
    bins = np.logspace(0, np.floor(np.log(degree_vals.max())) - 1, 100)
    hist = np.histogram(degrees, bins=bins)
    dist = scipy.stats.rv_histogram(hist)
    
    # Linear binning
    data_pdf = degree_counts/np.sum(degree_counts)
    data_cdf = np.cumsum(data_pdf)

    axes[0][0].scatter(degree_vals, data_pdf, label='PDF', facecolors='none', edgecolors='b', marker='.')
    axes[0][0].set_title('Linear PDF')
    axes[0][0].set_xlabel('k')
    axes[0][0].set_ylabel('p(k)')
    axes[0][0].set_xscale('log')
    axes[0][0].set_yscale('log')

    axes[0][1].scatter(degree_vals, 1 - dist.cdf(degree_vals), label='1 - CDF', facecolors='none', edgecolors='b', marker='.')
    axes[0][1].set_title('Log binning, cumulative')
    axes[0][1].set_xlabel('k')
    axes[0][1].set_ylabel('1 - P(k)')
    axes[0][1].set_xscale('log')
    axes[0][1].set_yscale('log')

    axes[1][0].plot(degree_vals, data_cdf, label='CDF', drawstyle='steps-post')
    axes[1][0].set_title('Linear CDF')
    axes[1][0].set_xlabel('k')
    axes[1][0].set_ylabel('P(k)')
    axes[1][0].set_xscale('log')
    axes[1][0].set_yscale('log')

    axes[1][1].plot(degree_vals, data_cdf, label='CDF', drawstyle='steps-post')
    axes[1][1].set_title('Log binning CDF')
    axes[1][1].set_xlabel('k')
    axes[1][1].set_ylabel('P(k)')
    axes[1][1].set_xscale('log')
    axes[1][1].set_yscale('log')

    plt.tight_layout()
    plt.show()


def get_betweenness(graph : nx.Graph, node_frac : float, seed : int):
    """
    Calculates the betweenness scores for the graph, potentially using a fraction of nodes
    for estimation

    :param graph: a nx.Graph or DiGraph instance
    :param node_frac: Float denoting the node fraction
    :param seed: Random seed to be used during the calculation
    """
    return nx.betweenness_centrality(G = graph, k = math.floor((graph.number_of_nodes() * node_frac)), seed=seed, weight='weight')

def get_hits(graph : nx.Graph | nx.DiGraph, tol : float):
    """
    Calculates the HITS Authority and Hub scores for the graph

    :param graph: a nx.Graph or DiGraph instance
    :parma tol: Tolerance for convergence of the calculation
    """
    return nx.hits(graph, tol = tol)

def get_pagerank(graph : nx.Graph | nx.DiGraph, alpha : float, tol : float):
    """
    Calclates the PageRank scores for the graph using networkx 

    :param graph: a nx.Graph or DiGraph instance
    :param alpha: Alpha to be used in the calculation
    :parma tol: Tolerance for convergence of the calculation
    """
    return nx.pagerank(graph, alpha, tol = tol)

def generate_power_law_clauset(n :int, gamma : float, k_min : int):
    """
    Generates power-law distributed data with the given parameters

    :param n: number of datapoints to generate
    :param gamma: Scaling parameter of the distribution
    :param k_min: K_min of the distribution

    :returns: np.array containing the generated data
    """
    # Source: Power-law distributions in empirical data, Aaron Clauset, Cosma Rohilla Shalizi, M. E. J. Newman, 2009, p. 39
    # 
    res = [] # holds the resulting numbers
    for _ in range(n):
        r = np.random.random()
        k_2 = k_min
        k_1 = k_2
        k_2 = 2*k_1
        # First part is the CDF of power law dist
        # Note that in the literature, the CDF is defined as P(x) = Pr(X >= x)
        while scipy.special.zeta(gamma, k_2)/ scipy.special.zeta(gamma, k_min) >= 1 - r:
            k_1 = k_2
            k_2 = 2*k_1
        
        # Using binary search, find the solution for some integer k, that is closest to 1-r
        # since the value of P(k_2) < P(k_1), this assignment is correct
        left = k_2
        right = k_1
        while True:
            mean = (left + right) / 2
            upper = scipy.special.zeta(gamma, np.floor(mean)) / scipy.special.zeta(gamma, k_min) # Floor has a higher P(k)
            lower = scipy.special.zeta(gamma, np.ceil(mean)) / scipy.special.zeta(gamma, k_min)

            # the real solution is somewhere between floor of the mean and ceil of the mean, then the floor
            # is our k
            if lower < 1 - r and upper > 1 - r:
                res.append(np.floor(mean))
                break

            if upper < 1 - r:
                # too low
                left = mean

            else:
                # too high
                right = mean

    return np.array(res)

def compute_metrics(args, graph : nx.Graph | nx.DiGraph):
    """
    Computes all relevant metrics for the given network.

    :param args: Command line arguments
    :param graph: A nx.Graph instance, for which the metrics will be computed

    :returns: A dataframe containing the computed metrics, indexed by nodes
    """
    pagerank = pd.DataFrame.from_dict(get_pagerank(graph, 0.85, 1e-8), orient='index', columns=['PageRank'])
    hits_hubs, hits_auth = get_hits(graph, 1e-6)
    hits_hubs = pd.DataFrame.from_dict(hits_hubs, orient='index', columns=['HubScore'])
    hits_auth = pd.DataFrame.from_dict(hits_auth, orient='index', columns=['AuthorityScore'])
    betweenness = pd.DataFrame.from_dict(get_betweenness(graph, args.node_frac, args.seed), orient='index', columns=['BetweennessCentrality'])

    # Join all of the dataframes together, indexed by node names
    result_df = pagerank.join(hits_hubs).join(hits_auth).join(betweenness)
    return result_df

def save_results(args, results : pd.DataFrame):
    """
    Saves the results in a file
    :param args: Command line arguments
    :param results: DataFrame to be saved
    """
    name = pathlib.PurePath(args.path).stem
    results.to_json(f"{name}_metrics.json")

def main(args):
    np.random.seed(args.seed)

    # Load graph
    df = pd.read_csv(args.path)
    graph = nx.from_pandas_edgelist(df, edge_attr = "weight", create_using = nx.DiGraph)
    # degrees = generate_power_law_clauset(50000, 2.5, 5)
    # plot_dist(graph)

    # Compute the desired metrics
    result_df = compute_metrics(args, graph)
    save_results(args, result_df)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)