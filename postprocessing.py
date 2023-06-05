"""
This script handles the postprocessing of the Twitter dataset, 
namely fetching user handles from their IDs, creating a networkx
Graph out of this dataset, and saving the graph in various formats.
"""
import time
import networkx as nx 
import pandas as pd
import tweepy
import json

INFO_PATH = 'twitter_info.json'
TWITTER_METRICS_PATH = 'twitter_network_metrics.json'
TWITTER_METRICS_NAMES_PATH = 'twitter_network_metrics_names.json'

def format_graph(graph : nx.Graph | nx.DiGraph):
    """
    Relabels all nodes to their string representation

    :param graph: nx.Graph or DiGraph to be modified
    """
    return nx.relabel_nodes(graph, lambda x : str(x))

def save_csv_metadata(df : pd.DataFrame, index=None, path='.\\Networks\\twitter_network_metadata.csv'):
    """
    Makes metadata for cosmograph
    in a csv format. If index colum is provided, 
    DataFrame index will be set to that column.
    WARNING: Old index will be dropped.
    """
    if index is not None:
        df = df.index(index)
        df.index.name = 'id'
    df.to_csv(path)

def save_csv(graph : nx.Graph, path='.\\Networks\\twitter_network_weighted.csv'):
    """
    Saves the graph in the csv format. Also renames the index to 'id' before saving.

    :param graph: A nx.Graph or DiGraph instance
    :param path: Save path
    """
    df = nx.to_pandas_edgelist(G=graph)
    df.index.name = 'id'
    df.to_csv(path)

def save_graphml(graph : nx.Graph | nx.DiGraph, path='.\\Networks\\twitter_network_with_attr.graphml'):
    """
    Saves the graph in the graphml format.

    :param graph: A nx.Graph or DiGraph instance
    :param path: Save path
    """
    nx.write_graphml(G=graph, path=path, encoding='utf-8', infer_numeric_types=True)

def rename_nodes(graph : nx.Graph | nx.DiGraph, info : pd.DataFrame, col='Username'):
    """
    Renames the nodes of the graph according to info dataframe

    :param graph: nx.Graph or DiGraph to be modified
    :param info: A dataframe containing the relabel data
    :param col: The column in which new labels are stored
    """
    return nx.relabel_nodes(graph, lambda x: info.loc[x][col])

def add_attributes(graph : nx.Graph | nx.DiGraph, df : pd.DataFrame):
    """
    Adds attributes from df to the nodes of graph.

    :param graph: nx.Graph or DiGraph to be modified
    :param df: DataFrame indexed by the nodes of the graph, containing attributes to be added
    """
    df = pd.read_json('twitter_network_metrics_names.json')
    
    for node in graph.nodes:
        info = df.loc[node]
        for col in info.index:
            if not isinstance(info[col], (str, int, float)):
                raise ValueError(f'Node missing a {col} value')
            graph.nodes[node][col] = info[col]

def resolve_users(to_resolve : list, names : dict, twitter_client : tweepy.Client, could_not_resolve : int):
    """
    Resolves the users in the to_resolve list, modifiying the names dictionary with 
    the resolved info under keys 'Username' and 'Name'. Each id that could not be resolved
    increments the could_not_resolve counter by one

    :param to_resolve: List of ids to resolve
    :param names: Dictionary in which to save the results
    :param twitter_client: tweepy.Client instance
    :param could_not_resolve: Counter for the number of unresolved ids

    :returns: New count of unresolved ids
    """
    try:
        response = twitter_client.get_users(ids=to_resolve)
        for user_info in response.data:
            names[str(user_info['id'])] = {}
            names[str(user_info['id'])]['Username'] = user_info['username']
            names[str(user_info['id'])]['Name'] = user_info['name']

        for err_info in response.errors:
            could_not_resolve +=1
            names[str(err_info['resource_id'])] = {}
            names[str(err_info['resource_id'])]['Username'] = 'Unknown#' + str(could_not_resolve)
            names[str(err_info['resource_id'])]['Name'] = 'Unknown#' + str(could_not_resolve)

    except tweepy.HTTPException:
        print('Too many requsests, sleeping for 15 min')
        time.sleep(15*60)

    return could_not_resolve

def twitter_id_to_name(df : pd.DataFrame):
    """
    Resolves the IDs in the twitter dataframe into usernames.
    Includes waiting for API responses and pagination.

    :param df: Dataframe indexed by user IDs that will be resolved
    :returns: Modified dataframe with a 'Username' column, containing the username linked to this ID, and
    'Name' column, containing the current display name of the user
    """
    # Start the Twitter Client
    with open(INFO_PATH, 'r') as file:
        auth_dict = json.load(file)
        bearer = auth_dict['bearer']
    twitter_client = tweepy.Client(bearer_token=bearer)
    
    names = {}
    to_resolve = []
    could_not_resolve = 0
    count = 0
    for user in df.index:
        if count < 100:
            if user.isdigit():
                to_resolve.append(user)
                count += 1
            else:
                names[user] = {}
                names[user]['Username'] = user
                names[user]['Name'] = user

        else:
            could_not_resolve = resolve_users(to_resolve, names, twitter_client, could_not_resolve)
            to_resolve = []
            count = 0
            if user.isdigit():
                to_resolve.append(user)
                count +=1
            else:
                names[user] = {}
                names[user]['Username'] = user
                names[user]['Name'] = user

    if len(to_resolve) > 0:
        resolve_users(to_resolve, names, twitter_client, could_not_resolve)

    df = df.join(pd.DataFrame.from_dict(data=names, orient='index'))
    df.to_json('twitter_network_metrics_names.json', indent=4)
    return df

def main(args):
    # df = pd.read_json(TWITTER_METRICS_PATH)
    # twitter_id_to_name(df)

    # df = pd.read_json(TWITTER_METRICS_NAMES_PATH)
    # g = nx.read_gpickle(args.path)
    # g = format_graph(g)
    # add_attributes(g, df)
    # save_edgelist(g)

    g = nx.readwrite.edgelist.read_weighted_edgelist('.\\Networks\\twitter_network.csv')
    df = pd.read_json(TWITTER_METRICS_NAMES_PATH)

    g = rename_nodes(g, df)
    save_csv_metadata(df, index='Username')
    save_csv(g, path='.\\Networks\\twitter_network_names.csv')

if __name__ == '__main__':
    main()