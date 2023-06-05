"""
This script handles data scraping from various sources, and saving it in a readable format.
"""

import praw
import json
import argparse
import pandas as pd
import datetime as dt
import snscrape.modules.twitter as sntwitter
from psaw import PushshiftAPI
from os import path

REDDIT_CLIENT_ID = "BpU3gU0TUYg2Kj_c6QPVPg"
REDDIT_SECRET_KEY = "jpMFV1SmlY9IBHtQZmBn5UhEOWdNWg"  # not so secret
STORAGE_FOLDER_NAME = "Data"
REDDIT_DATA_FILENAME = "reddit_new.json"
VERSION = "0.0.1"

parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="reddit", type=str, help="What data to scrape")
parser.add_argument("--login_path", default="pw.txt", type=str, help="Location of the file containing login data")
parser.add_argument("--data", default=".\\Data\\reddit_new.json", type=str, help="Path to post data")
parser.add_argument("--since", default="2021-10-29", type=str, help="Start date for scraping, in format YYYY-MM-DD")
parser.add_argument("--until", default="2022-02-08", type=str,
                    help="End date for scraping, in format YYYY-MM-DD")
parser.add_argument("--query", default=str("gme|gamestop"), type=str, help="Search query")


def get_tweets(args) -> pd.DataFrame:
    """
    Loads the tweets from snscrape and saves them in a dataframe. Timeframes are dependend on the argumnets from the CL.

    :param args: Command line arguments
    :return: DataFrame containg the tweets with the following info: Date, UserID, Content, ID, LikeCount, ReplyCount, RetweetCount,
    QuotedTweet, RetweetedTweet
    """

    # Dictionary which will later be turned into the DataFrame
    tweets = {
        'Date': [],
        'UserID': [],
        'Content': [],
        'ID': [],
        'LikeCount': [],
        'ReplyCount': [],
        'RetweetCount': [],
        'QuotedTweet': [],
        'RetweetedTweet': [],
    }

    # Tweet objects gathered as a response from snsscrape
    raw_tweets = sntwitter.TwitterSearchScraper(
        "{} since:{} until:{}".format(args.query, args.since, args.until)).get_items()

    # Save the tweed data to the dictionary
    for tweet in raw_tweets:

        tweets['Date'].append(tweet.date)
        tweets['UserID'].append(tweet.user.id)
        tweets['ID'].append(tweet.id)
        tweets['Content'].append(tweet.content)
        tweets['LikeCount'].append(tweet.likeCount)
        tweets['ReplyCount'].append(tweet.replyCount)
        tweets['RetweetCount'].append(tweet.retweetCount)

        # If it has a retweeted Tweet, save it's ID
        if tweet.retweetedTweet is not None:
            tweets['RetweetedTweet'].append(tweet.retweetedTweet.id)

        else:
            tweets['RetweetedTweet'].append(None)

        # If it quotes a tweet, save it's ID
        if tweet.quotedTweet is not None:
            tweets['QuotedTweet'].append(tweet.quotedTweet.id)

        else:
            tweets['QuotedTweet'].append(None)

    return pd.DataFrame(tweets)


def serialize_tweets(args, tweets: pd.DataFrame):
    """
    Saves the tweets to the given data path from args

    :param args: Command line arguments
    :param tweets: DataFrame containing the tweets to save
    """
    tweets.to_json(args.data)

def start_reddit_api():
    """
    Sets up the PRAW API which is later used for scraping. Uses the reddit client ID and secret
    provided in the 

    :return: PSAW API instance
    """
    pwd = str()
    with open('pw.txt', 'r') as file:
        pwd = file.read()

    # Create the praw api instance
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET_KEY,
        password=pwd,
        user_agent="WSB_SCRAPER/{}".format(VERSION),
        username="ml_project"
    )

    # Pass it to the Pushshift.io
    api = PushshiftAPI(reddit)
    return api


def get_historic_posts(api, search_string, subreddit, start_date=int(dt.datetime(2020, 1, 1).timestamp()),
                       end_date=int(dt.datetime(2021, 1, 7).timestamp())):
    """
    Gathers historic posts from the PushShift.io API, and returns them as a list

    :param api: PSAW API instance
    :param search_string: String used as a filter for the API
    :param subreddit: Name of the subreddit to scrape
    :param start_date: Datetime object converted to an int. Represents the starting date for the search filter
    :param end_date: Datetime object converted to an int. Represents the end date for the search filter

    :return: List containing the submissions found
    """
    result = api.search_submissions(after=start_date,
                                    before=end_date,
                                    subreddit=subreddit,
                                    is_self=True,
                                    user_removed=False,
                                    mod_removed=False,
                                    q=search_string)

    return list(result)


# Saves the submissions in a json, while storing relevant fields
def serialize_submissions(submissions):
    """
    Serializes the submissions into a json, while saving relevant fields. 
    Submissions are indexed by their ID.

    :param submissions: List of reddit submissions from PSAW API
    """
    out_dict = dict()
    destination = path.join('.', STORAGE_FOLDER_NAME, REDDIT_DATA_FILENAME)
    count = 0
    for post in submissions:
        if not (post.selftext == '[removed]' or post.selftext == '[deleted]'):
            date = dt.date.fromtimestamp(float(post.created_utc))

            # Remember some useful data about the post
            out_dict[post.id] = {
                'score': post.score,
                'selftext': post.selftext,
                'title': post.title,
                'created_utc': post.created_utc,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
            }

            count += 1

            # Data for the author might not be available.
            try:
                name = post.author.name
                id = post.author.id

                author = {
                    'name': name,
                    'id': id,
                }
                out_dict[post.id]['author'] = author
            except:
                # Couldn't find the author, set it to None
                out_dict[post.id]['author'] = None

    with open(destination, 'w') as file:
        json.dump(out_dict, file, indent=4)

    print("Saved {} Reddit posts".format(count))

def get_datetime(string):
    return dt.datetime.strptime(string,'%Y-%m-%d')

def main(args):
    if args.mode == "reddit":
        api = start_reddit_api()
        start_date = int(get_datetime(args.since).timestamp())
        end_date = int(get_datetime(args.until).timestamp())
        # Gathers all text posts which contain keywords gme or gamestop from wallstreetbets
        res = get_historic_posts(api, args.query, 'wallstreetbets', start_date=start_date, end_date=end_date)

        # Saves them in a json
        serialize_submissions(res)

    if args.mode == "twitter":
        tweets = get_tweets(args)
        serialize_tweets(args, tweets)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
