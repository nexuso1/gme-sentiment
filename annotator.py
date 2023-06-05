"""
This script provides a CLI for manual data annotation.

For usage, run this script from the terminal with the --help command
"""

import json
import pandas as pd
import numpy as np
import argparse
import re
from os import path

parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="reddit", type=str, help="Which data to annotate")

DATA_FOLDER = "Data"

def format_json(json_file):
    """
    Removes the 'irrelevant' posts from the file

    :param json_file: json dictionary
    :return: Cleaned dictionary
    """
    json_file.pop('irrelevant')
    return json_file

def highlight_gme(text):
    res = re.sub("gme|gamestop", '<<<<<<<<<<<<<GME>>>>>>>>>>>>>>>', text, flags=re.IGNORECASE)

    return res


# Pauses annotation by dumping the dict into a json
def save_progress(res: dict, dest_irr: str, dest_data: str):
    """
    Pauses annotation by dumping the dict into a json

    :param res: Dictionary with annotations
    :param dest_irr: File path to save the dictionary with irrelevant posts included
    :param dest_data: File path to save the dictionary without irrelevant posts
    """
    with open(dest_irr, 'w') as file:
        json.dump(res, file, indent=4)
        file.close()

    with open(dest_data, 'w') as file:
        json.dump(format_json(res), file, indent=4)
        file.close()


def annotate_twitter(filename: str):
    """
    Go through all the posts and remember the answer given by the user
    Commands:
    '1' = positive, '0' = negative, '3' - irrelevant, 'pause' - save progress
    '4' = skip post

    :param filename: Name of the Twitter dataset location
    """
    posts = pd.read_json(path.join(DATA_FOLDER, filename))
    dest_irr = path.join(DATA_FOLDER, 'twitter_annotated_with_irrelevant.json')
    dest_data = path.join(DATA_FOLDER, 'twitter_annotated.json')
    res = {}

    # Resume if not done
    if path.isfile(dest_irr):
        with open(dest_irr, 'r') as file:
            res = json.load(file)

        file.close()

    count = len(res.keys())
    negative_delta = 0

    # Make a sub-dictionary so that irrelevant posts are skipped
    if 'irrelevant' not in res.keys():
        res['irrelevant'] = {}

    post_order = np.random.permutation(posts.shape[0])
    for idx in post_order:

        if str(posts.iloc[idx]['ID']) in res:
            continue

        if str(posts.iloc[idx]['ID']) in res['irrelevant']:
            continue

        print()
        print(f'Total annotated:{count}')
        print(f'Negative delta: {negative_delta}')
        print("text: " + posts.iloc[idx]['Content'])
        value = input("Positive/Negative/Irrelevant/Pause/Skip = 1/0/3/pause/4 : ")
        if value == "3":
            if not str(posts.iloc[idx]['ID']) in res['irrelevant']:
                res['irrelevant'][str(posts.iloc[idx]['ID'])] = 1
            continue

        if value == "4":
            print("Skipped")
            continue

        if value == "pause":
            save_progress(res, dest_irr, dest_data)
            return

        if value =='0':
            negative_delta += 1
        count += 1
        res[str(posts.iloc[idx]['ID'])] = value
        print("====")
        print()
        print()

    save_progress(res, dest_irr, dest_data)


def annotate_reddit(filename: str):
    """
    Go through all the posts and remember the answer given by the user
    Commands:
    '1' = positive, '0' = negative, '3' - irrelevant, 'pause' - save progress

    :param filename: Name of the Reddit dataset location
    """
    filtered = 0
    with open(path.join(DATA_FOLDER, filename), 'r') as file:
        posts = json.load(file)
        dest_irr = path.join(DATA_FOLDER, "reddit_annotated_with_irrelevant.json")
        dest_data = path.join(DATA_FOLDER, 'reddit_annotated.json')
        res = {}

        # Resume if not done
        if path.isfile(dest_irr):
            with open(dest_irr, 'r') as file:
                res = json.load(file)

            file.close()

        count = len(res.keys())
        negative_delta = 0

        # Make a sub-dictionary so that irrelevant posts are skipped
        if 'irrelevant' not in res.keys():
            res['irrelevant'] = {}

        post_order = np.random.permutation(len(posts.keys()))
        for idx in post_order:
            post = list(posts.keys())[idx]

            if post in res:
                continue

            if post in res['irrelevant']:
                continue

            print()
            print(f'Total annotated:{count}')
            print(f'Negative delta: {negative_delta}')
            print("Filtered so far: {}".format(filtered))
            print("title:" + posts[post]['title'])
            print(f"text: {highlight_gme(posts[post]['selftext'])}")
            value = input(
                "Positive/Negative/Irrelevant/Skip/Pause/ = 1/0/3/4/pause : ")
            if value == "3":
                if post not in res['irrelevant']:
                    res['irrelevant'][post] = 1
                continue

            if value == "4":
                print("Skipped")
                continue

            if value == "pause":
                save_progress(res, dest_irr, dest_data)
                return

            if value =='0':
                negative_delta += 1

            count += 1
            res[post] = value
            print("====")
            print()
            print()

    save_progress(res, dest_irr, dest_data)


def main(args):
    if args.mode == 'reddit':
        annotate_reddit("reddit.json")

    if args.mode == 'twitter':
        annotate_twitter('twitter.json')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
