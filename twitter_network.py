"""
This script handles the creation of a Twitter network from the Twitter dataset, together 
with gathering additional data about retweets and followers, used to create edges and nodes
in the network.

For usage, run this script from the terminal with the --help command
"""

import networkx as nx
import pandas as pd
import argparse
import tweepy
import tweepy.errors
import json
import os
import time
import re

INFO_PATH = 'twitter_info.json' # Twitter credentials path
STATUS_PATH = '.\\Networks\\twitter_network_processing.STATUS' # Default status path
NETWORK_PATH = '.\\Networks\\twitter_network.gz' # Default network path

parser = argparse.ArgumentParser()
parser.add_argument("--twitter_path", default=".\\Data\\twitter.json", type=str, help="Path to the Twitter dataset")
parser.add_argument("--out", default=".\\Networks\\twitter_network.json", type=str, help="Output path")

# Setup the twitter client
with open(INFO_PATH, 'r') as file:
    auth_dict = json.load(file)
    bearer = auth_dict['bearer']

twitter_client = tweepy.Client(bearer_token=bearer)

def save_graph(graph : nx.DiGraph | nx.Graph):
    """
    Saves the graph during Tweet processing to the NETWORK_PATH

    :param graph: A nx.Graph or mx.DiGraph instance to be saved
    """
    if not os.path.exists(os.path.dirname(NETWORK_PATH)):
        os.makedirs(os.path.dirname(NETWORK_PATH))
    
    nx.readwrite.write_gpickle(graph, NETWORK_PATH)

    print("Graph saved. Status: {} nodes, {} edges".format(graph.number_of_nodes(), graph.number_of_edges()))

def save_progress(graph : nx.DiGraph | nx.Graph, count : int):
    """
    Saves progress during tweet processing. Saves the number
    of processed tweets in the STATUS_PATH file

    :param graph: A nx.DiGraph or nx.Graph instance
    :param count: How many tweets have been processed
    """
    if not os.path.exists(STATUS_PATH):
        os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    
    with open(STATUS_PATH, 'w') as file:
        file.write(str(count))
    
    save_graph(graph)
    print("Processed a total of {} tweets.".format(count))


# Adds (user, origin_user) edges to the graph
def add_users_from_list(graph : nx.DiGraph, origin_user_id : int, users : list):
    """
    Adds (user, origin_user_id) edges to the graph, created from the list of users

    :param graph: A nx.Digraph instance
    :param origin_user_id: ID of the target user
    :param users: List of source user IDs
    """
    for user in users:    
        if not graph.has_edge(user.id, origin_user_id):
            graph.add_edge(user.id, origin_user_id)
            graph[user.id][origin_user_id]['weight'] = 0

        graph[user.id][origin_user_id]['weight'] += 1

# Adds all followers of a user to the graph. Handles the communication with the Twitter API
# and waits until API allows it to make all requests
def add_followers(graph : nx.DiGraph, user_id : int, twitter_client : tweepy.Client):
    """
    Adds all followers of a user to the graph. Handles the communication with the Twitter API
    and waits until API allows it to make all requests.
    :param graph: A nx.Digraph instance
    :param user_id: ID of the author of the tweet
    :param twitter_client: A tweepy.Client instance
    """
    try:
        response = twitter_client.get_users_followers(user_id, max_results=1000)
    except tweepy.TooManyRequests:
        time.sleep(900) # 15 requests per 15 mins max
        response = twitter_client.get_users_followers(user_id, max_results=1000)
        
    add_users_from_list(graph, user_id, response.data)

    # We have multiple pages in the pagination
    if 'next_token' in response.meta:
        next_token = response.meta['next_token']
        while next_token is not None:
            try:
                response = twitter_client.get_users_followers(user_id, max_results=1000, pagination_token=next_token)
            except tweepy.TooManyRequests:
                time.sleep(900)
                response = twitter_client.get_users_followers(user_id, max_results=1000, pagination_token=next_token)

            add_users_from_list(graph, user_id, response.data)

            next_token = None
            if 'next_token' in response.meta:
                next_token = response.meta['next_token']

def add_retweeters(graph : nx.DiGraph, tweet_id : int, user_id : int, twitter_client : tweepy.Client):
    """
    Adds all retweeters of a given Tweet to the graph (via the Twitter API).Waits until API allows it to make all requests
    
    :param graph: a nx.Digraph instance
    :param tweet_id: ID of the Tweet for which to find retweeters
    :param user_id: ID of the author of the Tweet
    :param twitter_client: a tweepy.Client instance
    """

    try:
        response = twitter_client.get_retweeters(tweet_id, max_results=100)
    except tweepy.TooManyRequests:
        # Wait for the next period for requests
        time.sleep(900)
        response = twitter_client.get_retweeters(tweet_id, max_results=100)

    if response.data is not None:
        add_users_from_list(graph, user_id, response.data)

    # We have multiple pages in the pagination
    if 'next_token' in response.meta:
        next_token = response.meta['next_token']
        while next_token is not None:
            try:
                response = twitter_client.get_retweeters(tweet_id, max_results=100, pagination_token=next_token)
            except tweepy.TooManyRequests:
                time.sleep(900)
                response = twitter_client.get_retweeters(tweet_id, max_results=100, pagination_token=next_token)

            if response.data is not None:
                add_users_from_list(graph, user_id, response.data)

            next_token = None
            if 'next_token' in response.meta:
                next_token = response.meta['next_token']

def get_mentions(text : str, twitter_client : tweepy.Client):
    """
    Finds all mentioned users in the text and returns them in a list
    
    :param text: String from which to extract the users
    :param twitter_client: a tweepy.Client instance
    """
    usernames = re.findall('@(\w+)', text)
    res = []
    for user in usernames:
        try:
            response = twitter_client.get_user(username=user).data # Response is None or User
        except tweepy.errors.BadRequest:
            # Seems to be a problem with the character limit
            # Assuming this is not very frequent, we dont need to 
            # use their ID
                res.append(user)
                continue

        if response is not None:
            res.append(response.id)

    return res

def add_mentioned(graph : nx.DiGraph, tweet, origin_user_id : int):
    """
    Adds edges (origin_user, mentioned) from the tweet to the graph

    :param graph: A networkx graph instance
    :param tweet: Tweet as a pd.Series insatnce
    :param origin_user_id: ID of the author of the Tweet
    """
    mentions = get_mentions(tweet['Content'], twitter_client)
    for mentioned in mentions:
        if not graph.has_edge(origin_user_id, mentioned):
            graph.add_edge(origin_user_id, mentioned)
            graph[origin_user_id][mentioned]['weight'] = 0

        graph[origin_user_id][mentioned]['weight'] += 1

def create_graph(args):
    """
    Main function that runs the network creation algorithm. 
    For every Tweet, it adds the author, his followers, 
    people who retweeted the tweet, and people mentioned in the Tweet
    to the graph and connects them. Also updates the weight of edges if 
    an edge would be repeated. Periodically saves the progress in a file.

    :param args: Command line arguments
    """
    tweets = pd.read_json(args.twitter_path)
    tweets['Date'] = tweets['Date'].apply(lambda x: x.date())
    tweets = tweets.sort_values('Date')
    graph = nx.DiGraph()
    if os.path.exists(NETWORK_PATH):
        graph = nx.readwrite.read_gpickle(NETWORK_PATH)

    start_idx = 0 # Index of last processed tweet in the previous session
    count = 0 # Number of tweets processed since last reset
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, 'r') as file:
            start_idx = int(file.readline())

    for i in range(start_idx, tweets.shape[0]):
        # Save progress every 10 tweets
        if count == 10:
            save_progress(graph, start_idx)
            count = 0
        
        tweet = tweets.iloc[i]

        # Add followers of the author to the graph, and link them with him
        if not graph.has_node(tweet['UserID']):
            graph.add_node(tweet['UserID'])
            add_followers(graph, tweet['UserID'], twitter_client)

        # Add people who retweeted the tweet, and link them to author
        add_retweeters(graph, tweet['ID'], tweet['UserID'], twitter_client)

        # Add mentioned users
        add_mentioned(graph, tweet, tweet['UserID'])


        start_idx += 1
        count += 1

    save_progress(graph, start_idx)


def main(args):
    create_graph(args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)