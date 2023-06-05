"""
This script handles sentiment prediction on the whole dataset, with 
a given model as an input. Also contains functions for sentiment 
visualization, as well as interacting with the model.
"""
import sys
from os.path import join, abspath
# add Models folder to the path, so that file is in the Models package
models_path = join(abspath(''), 'Models')
sys.path.append(models_path)

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import lzma
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.colors

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default=".\\Data\\total_dataset.json", type=str, help="Path to total dataset")
parser.add_argument("--reddit_path", default='.\\Data\\reddit.json', type=str, help="Path to Reddit dataset")
parser.add_argument("--twitter_path", default='.\\Data\\twitter.json', type=str, help="Path to Twitter dataset")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--model_name", type=str, default='svc_best', help='Target model name (without .MODEL)')
parser.add_argument('--model_folder', type=str, default='Models', help='Folder containing the models, relative to this script')
parser.add_argument("--interactive", type=bool, default=False, help='Display a text input, which will be annotated by the model')

class RnnVectorizer:
    """
    A wrapper for the RNN vectorizer, includes a transform function that just
    calls the predict() function of the vectorizer
    """
    def __init__(self, vectorizer : tf.keras.Model) -> None:
        self.vectorizer = vectorizer

    def transform(self,text):
        return self.vectorizer.predict(text)
    
class KerasModelWrapper:
    """
    A wrapper for a keras model, that returns predictions as the argmax of its outputs.
    """
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, text):
        preds = self.model.predict(text)
        return np.argmax(preds, axis=1)

def plot_bar(fig, ax, x, y, cmap, log_norm=True, **kwargs):
    """
    Creates a bar plot on the ax in the figure, and adds a colorbar.
    The values are first normalized. Optionally, a logarithmic norm can be used.
    The norm object will be returned. Kwargs for the plotting can be included as well.

    :param fig: Matplotlib figure
    :param ax: Maptlotlib ax
    :param x: x axis data
    :param y: y axis data (to be normalized)
    :param cmap: matplotlib.cm.cmap object to be used
    :param log_norm: Bool, whether to use logarithmic norm or not

    :returns: The norm object
    """
    # Normalize the values, so that the lowest is 
    # the 25th quantile
    if log_norm:
        norm = matplotlib.colors.LogNorm(vmin=max(1, y.quantile(0.25)))
    else:
        norm = matplotlib.colors.Normalize()

    normalized_y = norm(y)

    # Get color codes
    colors = cmap(normalized_y)

    # Plot the bar
    ax.bar(x, y, color=colors, **kwargs)

    # Add a colorbar
    add_colorbar(fig, ax, norm, cmap)

    return norm

def add_colorbar(fig, ax, norm, cmap):
    """
    Creates a colorbar on figure fig's ax ax, 
    with norm and cmap

    :param fig: Matplotlib figure
    :param ax: Maptlotlib ax
    :param norm: A matplotlib norm applied to the data
    :param cmap: A matplotlib.cm.cmap instance used on the data
    """

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

def sentiment_score(period_data : pd.DataFrame, mode='both'):
    """
    Calculates the sentiment score given a dataframe containing
    posts with both sentiment and post score columns

    :param period_data: The mentioned dataframa
    :param mode: Whether to consider all posts, or only Tweets or Reddit posts. Available choices: 'both', 'reddit', 'twitter'

    :returns: The calculated sentiment score
    """
    if mode == 'reddit':
        period_data = period_data[period_data['source'] == 'reddit'] 
    
    if mode == 'twitter':
        period_data = period_data[period_data['source'] == 'twitter']

    grouped = period_data.groupby(by=period_data['sentiment']).agg({'score' : 'sum' })
    total = grouped.sum('index')
    positive = 0
    negative = 0
    if 'negative' in grouped.index.values:
        negative = grouped.loc['negative']

    if 'positive' in grouped.index.values:
        positive = grouped.loc['positive']

    return (positive - negative) / total

def plot_sentiment(data : pd.DataFrame, mode='both'):
    """
    Plots the sentiment from the data, assuming it already has a "score"
    column

    :param data: Dataframe containing the posts with their sentiment
    """
    grouped = data.groupby(data['datetime'].dt.date)
    plt.figure()
    scores = {}
    for group in grouped.groups:
        score = sentiment_score(grouped.get_group(group), mode)
        scores[group] = score

    cmap = matplotlib.cm.get_cmap('PiYG')

    df = pd.DataFrame.from_dict(scores, orient='index', columns = ['score'])
    df = df.set_index(pd.DatetimeIndex(df.index))
    df = df.reindex(pd.date_range(df.index[0], df.index[-1]), fill_value=0)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 10)

    plot_bar(fig, ax, df.index, df['score'], log_norm=False, cmap=cmap)
    plt.show()

def get_predictions(model, preprocess, data):
    """
    Transforms the data using the preprocess instance and makes predicitons 
    using the model instance

    :param model: A model instance implementing a predict(data) function
    :param preprocess: A preprocessing transformer instance implementing a transform(data) function
    :param data: Data to predict

    :returns: An array of predictions in the same order as data
    """
    preprocessed = preprocess.transform(data['text'])
    try: 
        predictions = model.predict(preprocessed)
    except ValueError:
        preprocessed = preprocessed.toarray()
        model.predict(preprocessed)

    return predictions

def sentiment_to_label(sentiment):
    """
    Helper function that turns the numeric sentiment labels into strings

    :param sentiment: Int representing the sentiment
    :returns: A textual representation of the sentiment
    """
    if sentiment == 0:
        return 'negative'
    if sentiment == 1:
        return 'positive'
    if sentiment == 2:
        return 'neutral'

    raise ValueError(f'Invalid sentiment passed ({sentiment})')

def calculate_sentiment(model, preprocess, total_dataset : pd.DataFrame):
    """
    Estimates the sentiment for all posts in the total_dataset, using the given 
    model and preprocess function

    :param model: A model instance implementing a predict(data) function
    :param preprocess: A preprocessing transformer instance implementing a transform(data) function
    :returns: Total dataset with a new column called "sentiment" added, that contains the predicted sentiment
    """
    preds = get_predictions(model, preprocess, total_dataset)
    total_dataset = total_dataset.join(pd.DataFrame(preds, columns=['sentiment']))
    total_dataset['sentiment'] = total_dataset['sentiment'].apply(sentiment_to_label)
    return total_dataset

def text_input(model, preprocess):
    """
    Console input handler for the interactive mode

    :param model: Loaded model, has to implement a predict(data) function
    :param preprocess: Preprocessor for this model, has to implement a transform(data) function
    """
    while True:
        text = input('Input your text here: ')
        if text == 'exit':
            return

        text = preprocess.transform([text])
        prediction = model.predict(text)

        print(f'Model output: {sentiment_to_label(prediction)}')


def load_rnn(model_folder, name):
    """
    Loads the RNN model, which has to be loaded via tensorflow. Also wraps the 
    vectorizer in a wrapper

    :param model_folder: Name of the model folder
    :param name: Model name (without.MODEL)
    """
    model_path = os.path.join('.', model_folder, f'{name}.MODEL')
    model = tf.keras.models.load_model(model_path, compile=False)
    vectorizer_path = os.path.join('.', model_folder, f'{name}_vectorizer.MODEL')
    vectorizer = tf.keras.models.load_model(vectorizer_path, compile=False)
    return KerasModelWrapper(model), RnnVectorizer(vectorizer)

def load_nn(model_folder, name):
    """
    Loads the NN model, which has to be loaded via tensorflow.

    :param model_folder: Name of the model folder
    :param name: Model name (without.MODEL)
    """
    model_path = os.path.join('.', model_folder, f'{name}.MODEL', 'model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    vectorizer_path = os.path.join('.', model_folder, f'{name}.MODEL', 'preprocess.PICKLE')
    with lzma.open(vectorizer_path, 'r') as f:
        preprocess = pickle.load(f)

    return KerasModelWrapper(model), preprocess

def load_model(model_folder : str, name : str):
    """
    Loads the specified model
    :param model_folder: Name of the model folder (assumed to be in the directory of this script)
    :param name: Model name (without .MODEL)
    :returns: A tuple of (model, preprocessor)
    """
    try:
        model_path = os.path.join('.', model_folder, f'{name}.MODEL')
        with lzma.open(model_path, 'r') as file:
            model, preprocess = pickle.load(file)

        return model, preprocess
    except (FileNotFoundError, PermissionError):
        try:
            return load_rnn(model_folder, name)
        except (IOError, FileNotFoundError, PermissionError, ):
            return load_nn(model_folder, name)

def reddit_score(x, reddit):
    """
    Returns the post score of a Reddit post. In this case, it is the
    number of upvotes * upvote ratio

    :param x: A pd.Series representing the reddit post
    :param reddit: A pd.DataFrame containg all Reddit posts and their information
    """
    return reddit.loc[x['id']]['score'] * reddit.loc[x['id']]['upvote_ratio']

def twitter_score(x, twitter):
    """
    Returns the post score of a Tweet. In this case, it is the like count.

    :param x: A pd.Series representing the Tweet
    :param twitter: A pd.DataFrame containing all Tweets and their information
    """
    return twitter.loc[x['id']]['LikeCount']

def add_scores(total_dataset :pd.DataFrame, reddit_path, twitter_path):
    """
    Add post scores to posts in the total_dataset

    :param total_dataset: The dataset containing both reddit posts and tweets (at least one of each)
    :param reddit_path: Path to the Reddit dataset
    :param twitter_path: Path to the Twitter dataset
    """
    reddit = pd.read_json(reddit_path, orient='index')
    twitter = pd.read_json(twitter_path)
    
    reddit_mask = total_dataset['type'] == 'reddit_post'
    twitter_mask = total_dataset['type'] == 'tweet'

    total_dataset['score'] = 0
    total_dataset.loc[reddit_mask, 'score'] = total_dataset[reddit_mask].apply(reddit_score, axis=1, args=[reddit])
    total_dataset.loc[twitter_mask, 'score'] = total_dataset[twitter_mask].apply(twitter_score, axis=1, args=[twitter])

    return total_dataset

def main(args):
    # Interactive mode
    if args.interactive:
        model, preprocess = load_model(args.model_folder, args.model_name)
        text_input(model, preprocess)
        return
    
    # Calculate the sentiment for the whole dataset and plot it as a bar plot
    dataset = pd.read_json(args.dataset_path)
    model, preprocess = load_model(args.model_folder, args.model_name)
    with_scores = add_scores(dataset, args.reddit_path, args.twitter_path)
    sentiment = calculate_sentiment(model, preprocess, with_scores)
    plot_sentiment(sentiment)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)