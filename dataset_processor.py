"""
This script handles processing data from Reddit and Twitter received from scraper.py, 
and saving it in a more unified and useful format.

For usage, run this script from the terminal with the --help command
"""

import json
import pandas as pd
import argparse

# This wouldn't have been needed if I'd saved the posts properly the first time, but oh well

parser = argparse.ArgumentParser()

parser.add_argument("--reddit", default=".\\Data\\reddit.json", type=str, help="Path to reddit data")
parser.add_argument("--twitter", default=".\\Data\\twitter.json", type=str, help="Path to twitter data")
parser.add_argument("--twitter_annotated", default=".\\Data\\twitter_annotated_with_irrelevant.json", type=str, help="Path to twitter data with irrelevant posts included")
parser.add_argument("--reddit_annotated", default=".\\Data\\reddit_annotated_with_irrelevant.json", type=str,
                    help="Path to dataset with irrelevant posts included")
parser.add_argument("--train_out", default=".\\Data\\train_dataset_new.json", type=str,
                    help="Path to output train dataset")
parser.add_argument("--total_out", default=".\\Data\\total_dataset_new.json", type=str,
                    help="Path to output total dataset (all posts)")
parser.add_argument("--alt_out", default=".\\Data\\train_dataset_alt_new.json", type=str,
                    help="Path to output alt dataset (with irrelevant posts included)")


def create_alt_dataset(reddit_path: str, reddit_annotated_path: str, twitter_path : str, twitter_annotated_path : str):
    """
    Creates a dataset with posts marked as irrelevant aswell

    :param reddit_path: Path to the Reddit total dataset
    :param reddit_annotated_path: Path to Reddit Annotated dataset
    :param twitter_path: Path to the Twitter total dataset
    :param twitter_annotated_path: Path to Twitter Annotated datase
    """
    out_dataset = {}

    with open(reddit_path, 'r') as file:
        reddit_data = json.load(file)
        file.close()

    with open(reddit_annotated_path, 'r') as file:
        reddit_annotated = json.load(file)
        file.close()

    with open(twitter_path, 'r') as file:
        twitter_data = pd.read_json(file)
        file.close()

    with open(twitter_annotated_path, 'r') as file:
        twitter_annotated = json.load(file)
        file.close()

    idx = 0
    
    # Assign class 2 to irrelevant
    for post in reddit_annotated['irrelevant']:
        out_dataset[idx] = {}
        out_dataset[idx]['text'] = reddit_data[post]['title'] + ' ' + reddit_data[post]['selftext']
        out_dataset[idx]['type'] = 'reddit_post'
        out_dataset[idx]['id'] = post
        out_dataset[idx]['target'] = "2"
        idx += 1

    # index by ID
    twitter_data = twitter_data.set_index('ID')
    for post in twitter_annotated['irrelevant']:
        out_dataset[idx] = {}
        out_dataset[idx]['text'] = twitter_data.loc[int(post)]['Content']
        out_dataset[idx]['type'] = 'tweet'
        out_dataset[idx]['id'] = post
        out_dataset[idx]['target'] = "2"
        idx += 1

    reddit_annotated.pop('irrelevant')
    twitter_annotated.pop('irrelevant')

    for post in reddit_annotated:
        out_dataset[idx] = {}
        out_dataset[idx]['text'] = reddit_data[post]['title'] + ' ' + reddit_data[post]['selftext']
        out_dataset[idx]['type'] = 'reddit_post'
        out_dataset[idx]['id'] = post
        out_dataset[idx]['target'] = reddit_annotated[post]
        idx += 1

    for post in twitter_annotated:
        out_dataset[idx] = {}
        out_dataset[idx]['text'] = twitter_data.loc[int(post)]['Content']
        out_dataset[idx]['type'] = 'tweet'
        out_dataset[idx]['id'] = post
        out_dataset[idx]['target'] = twitter_annotated[post]
        idx += 1

    # Needs to be transposed so that the indexing makes more sense
    out = pd.DataFrame(out_dataset).T

    return out

def create_total_dataset(reddit_path, twitter_path):
    """
    Creates the total dataset, which contains all posts from reddit and
    twitter. Standardizes the timestamps to pd.datetime.
    Output dataset has the following columns:
        'type': Whether the post is a Tweet or Reddit post
        'text': Text of the post
        'id': id of the post (from Twitter or Reddit specifically)
        'datetime': When the post was created
    
    """
    reddit_data = pd.read_json(reddit_path)
    twitter_data = pd.read_json(twitter_path)

    total_dataset = {}
    idx = 0
    for post in reddit_data:
        total_dataset[idx] = {}
        total_dataset[idx]['type'] = 'reddit_post'
        total_dataset[idx]['text'] = reddit_data[post]['title'] + ' ' + reddit_data[post]['selftext']
        total_dataset[idx]['id'] = post
        total_dataset[idx]['datetime'] = pd.to_datetime(reddit_data[post]['created_utc'], unit='s')
        idx += 1

    for post in twitter_data.index:
        total_dataset[idx] = {}
        total_dataset[idx]['text'] = twitter_data.loc[int(post)]['Content']
        total_dataset[idx]['type'] = 'tweet'
        total_dataset[idx]['id'] = post
        total_dataset[idx]['datetime'] = twitter_data.loc[int(post)]['Date']
        idx += 1

    return pd.DataFrame(total_dataset).T

def create_datasets(reddit_path: str, reddit_annotated_path: str, twitter_path : str, twitter_annotated_path : str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a dataset with positive and negative posts. Also creates the total dataset,
    which combines all the data from twitter and reddit together into one file.

    :param reddit_path: Path to the Reddit total dataset
    :param reddit_annotated_path: Path to Reddit Annotated dataset
    :param twitter_path: Path to the Twitter total dataset
    :param twitter_annotated_path: Path to Twitter Annotated datase
    """
    train_dataset = {} # Initial dict for the train dataset
    total_dataset = {} # Initial dict for the total dataset
    with open(reddit_path, 'r') as file:
        reddit_data = json.load(file)
        file.close()

    with open(reddit_annotated_path, 'r') as file:
        reddit_annotated = json.load(file)
        file.close()

    with open(twitter_path, 'r') as file:
        twitter_data = pd.read_json(file)
        file.close()

    with open(twitter_annotated_path, 'r') as file:
        twitter_annotated = json.load(file)
        file.close()

    # Remove the irrelevant posts
    reddit_annotated.pop('irrelevant')
    twitter_annotated.pop('irrelevant')
    # First, create the training dataset (only annotated data)
    idx = 0
    for post in reddit_annotated:
        train_dataset[idx] = {}
        train_dataset[idx]['text'] = reddit_data[post]['title'] + ' ' + reddit_data[post]['selftext']
        train_dataset[idx]['type'] = 'reddit_post'
        train_dataset[idx]['id'] = post
        train_dataset[idx]['target'] = reddit_annotated[post]
        idx += 1

    twitter_data = twitter_data.set_index('ID')
    for post in twitter_annotated:
        train_dataset[idx] = {}
        train_dataset[idx]['text'] = twitter_data.loc[int(post)]['Content']
        train_dataset[idx]['type'] = 'tweet'
        train_dataset[idx]['id'] = post
        train_dataset[idx]['target'] = twitter_annotated[post]
        idx += 1

    # Need to be transposed so that the indexing makes more sense
    train = pd.DataFrame(train_dataset).T
    return train


def save_datasets(datasets : list, paths : list):
    """
    Utility function which saves the list of datasets

    :param datasets: List of datasets to save
    :param paths: Paths to save the dasets in
    """
    for i in range(len(datasets)):
        datasets[i].to_json(paths[i], indent=4)

def main(args):
    """
    Creates train, total and alt datasets and then saves them

    :param args: Command line arguments
    """
    train = create_datasets(args.reddit, args.reddit_annotated, args.twitter, args.twitter_annotated)
    total = create_total_dataset(args.reddit, args.twitter)
    alt = create_alt_dataset(args.reddit, args.reddit_annotated, args.twitter, args.twitter_annotated)
    save_datasets([train, total, alt], [args.train_out, args.total_out, args.alt_out])
    print("Datasets created.")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
