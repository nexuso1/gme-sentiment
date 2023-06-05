"""
This script handles the training of neural network models.

For usage, run this script from the terminal with the --help command
"""

import sklearn
import argparse
import pickle
import lzma
import pandas as pd
import sklearn.preprocessing
import sklearn.feature_extraction.text
import re
import tensorflow as tf
import numpy as np
import gc
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score
from tensorflow_addons.metrics import F1Score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default=os.path.join('Data', 'corrected_train_dataset_alt.json'), type=str, help="Path to annotated dataset")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.2, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--results_path", default="results_new.json", type=str, help="Path to results dataframe")
parser.add_argument("--model_folder", default=".\\Models\\", help="Model folder path")
parser.add_argument("--model_path", default="\\Models\\neural_network.MODEL")
parser.add_argument("--name", default='nerual_network_ensemble', help="Custom model name")
parser.add_argument("--save", default=True, type=bool, help='Whether or not to save the trained model in the Models folder.')
parser.add_argument("--force_dense", default=True, type=bool, help="Force dense representation of the feature matrix.")

def make_model(train_data, train_target, layer_shapes : tuple, activation : str, dropout : float):
    """
    Handles the NN creation with a customizable architecture and hyperparameters

    :param train_data: Training dataset
    :param test_data: Testing dataset
    :param train_target: Training target dataset
    :param test_target: Test target dataset
    :param layer_shapes: Tuple defining the shape of the NN, with the first index being the first layer
    :param activation: Name of the activation function used in the NN
    :param dropout: Dropout strength (Set to None if not desired)
    :return: Compiled model
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(train_data.shape[1],))

    for i, layer in enumerate(layer_shapes):
        model.add(tf.keras.layers.Dense(layer, activation=activation))
        if dropout is not None and i != len(layer_shapes) - 1:
            model.add(tf.keras.layers.Dropout(rate=dropout))
    
    model.add(tf.keras.layers.Dense(train_target.shape[1], activation='softmax'))

    return model

def compile_model(train_data, train_target, model : tf.keras.Model, learning_rate : float, 
                label_smoothing : float, epochs : int):
    """
    Compiles the model and sets the learning rate schedule

    :param train_data: Training examples
    :param train_target: Training labels
    :param model: Model to compile
    :param learning rate: Initial learning rate
    :param alpha: Minimal learning rate as a fraction of the initial LR
    :param weight_decay: Weight decay applied in the AdamW algorithm
    :param label_smoothing: Label smoothing applied to test labels
    :return: Compiled model
    """
    # Schedule
    learning_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * train_data.shape[0]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics = [tf.keras.metrics.CategoricalAccuracy(), F1Score(train_target.shape[1], average='macro')]
    )

    return model

def to_dense(x):
    """
    Helper function for preprocessing in a Pipeline. Turns the
    sparse matrix given to this function into a dense matrix

    :param x: Sparse matrix
    :return: Dense form of matrix x 
    """
    return x.toarray()

def preprocess(data : pd.DataFrame, min_df=5, ngram_range=(1,2), use_pca=False, replace_links=False, n_components=200, analyzer='word'):
    """
    Preprocess the text, replacing links with '[link]' and other
    special character removals, and transform it via tfidf vectorizer.
 
    :param data: Source Dataframe
    :param min_df: Minimum df for tfidf
    :param ngram_range: ngram size to consider when creating features
    :param analyzer: Whether to take into account word ngrams or character ngrams. Values can be 'word' or 'char'
    :param use_pca: Whether to use PCA or not
    :param replace_links: Whether to replace links or not
    :param n_components: Number of principal components to use for PCA
    
    :return: (Preprocessed text, tfidf vectorizer, vectorizer description)
    """

    # Replace links first
    if replace_links:
        for i in data.index:
            post = data.loc[i]
            data.at[i] = re.sub(pattern='http[s]*:\/\/[^\s]+', repl='[link]', string=post)

    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, analyzer=analyzer)
    pipe = Pipeline([
        ('tfidf', tfidf)
    ])

    desc = f'TF-IDF: {str(tfidf)}'
    if use_pca:
        pca = PCA(n_components=n_components, random_state=42)
        pipe.steps.append(('to_dense', FunctionTransformer(to_dense, accept_sparse=True)))
        pipe.steps.append(('pca', pca))
        features = pipe.fit_transform(data)

        desc = desc + f' PCA: {str(pca)}'
    else:
        pipe.steps.append(('to_denser', FunctionTransformer(to_dense, accept_sparse=True)))
        features = pipe.fit_transform(data)
    return features, pipe, desc

def training_wrapper(args, model_func, model_args : tuple, prep_args : tuple):
    """
    Training wrapper for interaction with GUI.
    Trains and saves the model, and returns a (model, transformer) tuple

    :param args: Command line args
    :param model_func: Function that will handle the training
    :param model_args: Arguments for the model training function
    :param prep_args: Arguments for the preprocess function
    :return: A tuple containing the model and transformer
    """
    # Create a log
    log = utils.Log(args.results_path, model_name='', desc='')
    
    args.name = 'neural_network_ensemble'

    # Train the model
    models, transformer = model_func(args, prep_args, log, *model_args)


    # Save the model if requested
    if args.save:
        save_model(args, models, transformer)

    return models, transformer

def save_model(args, model : tf.keras.Model, transformer):
    """
    Saves the trained model and transformer
    :param args: Command line arguments
    :param model: Model to save
    :param transformer: Input transformer for the model. Expected to be sklearn.Pipeline
    """
    model.save(f'{args.model_folder}/{args.name}.MODEL/model.h5')
    with lzma.open(f"{args.model_folder}/{args.name}.MODEL/preprocess.PICKLE", 'w') as file:
        pickle.dump(transformer, file)

    print('Model saved.')

def create_ensemble(input_shape, models : list[tf.keras.Model]):
    """
    Creates an ensemble model out of models inside the array 'models'.

    :param input_shape: Shape of the input into a singular model
    :param models: List of models to be included in the ensemble
    :return: Compiled ensemble model which averages the models outputs
    """
     # Create ensemble model
    inputs = tf.keras.layers.Input(shape=input_shape)
    model_layers = [m(inputs) for m in models]
    outputs = tf.keras.layers.average(model_layers)
    ensemble = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile it
    ensemble.compile()
    return ensemble

def neural_network_ensemble(args, preprocess_args : list, log : utils.Log, n_models = 10, sample_fraction = 1, 
        layer_shapes=(256,512), activation='relu', epochs=30, batch_size=32,
          learning_rate = 0.001, dropout=0.2, label_smoothing=0.1, patience = 10):
    """
    Neural Network Ensemble

    Creates and trains an ensemble of sequential neural networks with given layer shapes.

    :param args: For seed and test size purposes
    :param preprocess_args: Arguments package for the preprocess function
    :param data: Dataset on which to train the model
    :param n_models: Number of NNs in the ensemble
    :param sample_fraction: Fraction of data to use for subsampling during training of individual models
    :param layer_shapes: A tuple of layers shapes for the individual NNs. First number is the number of neurons in the first hidden layer, and so on.
    :param activation: What activation function to use. Can be a string, an instance of a keras loss or a custom function
    :param epochs: Number of training epochs
    :param batch_size: Batch size during training
    :param learning_rate: Learning rate used during training
    :param dropout: Dropout rate for the Dropout layers between hidden layers
    :param label_smoothing: Label smoothing rate applied to class labels during training
    :param patience: Patience to use for early stopping
    :return: A tuple of (Model, transformer)
    """
    # Set log name to neural network ensemble
    log.name = 'neural_network_ensemble'

    # Load the dataset
    df = pd.read_json(args.dataset_path)
    
    model, transformer = train_model(args, preprocess_args, df, log, learning_rate, n_models,
                                      sample_fraction, layer_shapes, activation, epochs, batch_size,
                                        dropout=dropout, label_smoothing=label_smoothing, patience=patience)
    return model, transformer

def train_model(args, preprocess_args : list, data : pd.DataFrame, log : utils.Log, learning_rate,
        n_models : int, sample_fraction : float, layer_shapes : tuple, activation : str,
          epochs : int, batch_size : int, dropout : float, label_smoothing : float,  patience : int, n_splits=5):

    """
    Trains the NN with the given parameters. Handles data preprocessing and model creation itself.

    :param args: Mostly for seed and test size purposes
    :param preprocess_args: Arguments for the preprocess function
    :param data: DataFrame containing the dataset
    :param result: DataFrame containing model results
    :param n_models: Number of models in the ensemble
    :param layer_shapes: Tuple defining the shape of the NN, with the first index being the first layer
    :param activation: Name of the activation function
    :param n_splits: How many splits to use for KFold cross-validation
    :param dropout: Dropout rate of the dropout layer between each hidden layer
    :param label_smoothing: Label smoothing rate applied to the class labels
    :param alpha:
    :return: A tuple of (Model, transformer)
    """

    # Use Stratified K-fold cross-validation to train the model
    stf = StratifiedKFold(n_splits, random_state=args.seed, shuffle=True)
    splits = stf.split(data.text, data.target)

    # Transform target to categorical features
    target = tf.keras.utils.to_categorical(data.target)

    progress = 1 # Training progress counter
    best_score = 0
    best_ensemble = None

    for train_idx, test_idx in splits:
        # Arrays that will hold individual models and their accuracy scores
        members, scores = [], []

        # Split the data into training and testing set
        train_data, test_data = data.text[train_idx], data.text[test_idx]
        train_target, test_target = target[train_idx], target[test_idx]

        resample_target = data.target[train_idx]

        # Preprocess data
        train_data, tfidf, transf_desc = preprocess(train_data, *preprocess_args)
        test_data_prep = tfidf.transform(test_data)

        for i in range(n_models):
            idxs = np.array(range(train_data.shape[0]))
            
            # Resample the dataset so that each NN has a bit different data to learn from
            train_idxs = resample(idxs, replace=True, n_samples=int(idxs.shape[0] * sample_fraction), random_state=args.seed, stratify=resample_target)

            # Create corresponding train datasets
            trainX, trainY = train_data[train_idxs], train_target[train_idxs]

            # Create and train the NN on the data
            model = make_model(trainX, trainY, layer_shapes, activation,dropout=dropout)
            model = compile_model(trainX, trainY, model, learning_rate, label_smoothing, epochs)
            
            # Make a callback for early stopping
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=False, mode='min')

            model.fit(trainX, trainY, validation_data=(test_data_prep, test_target), epochs=epochs,
                       batch_size=batch_size, callbacks=[callback], verbose=1)
            _, acc, f1 = model.evaluate(test_data_prep, test_target, verbose=0)

            # Save its results
            members.append(model)
            scores.append(acc)
            print("Model {} had an accuracy of {:.4f} and F1 score of {:.4f}".format(i, acc, f1))

        gc.collect()
        
        # Create the ensemble
        ensemble = create_ensemble(train_data.shape[1:], members)

        # Get predictions of the ensemble
        preds = ensemble.predict(tfidf.transform(test_data))

        # Format them so that they're compatible with sklearn metrics
        preds_formatted = tf.argmax(preds, axis=1)

        # Calculate metrics
        log.evaluate_metrics(data.target[test_idx], preds_formatted)

        # If the F1 score is better than the current best, save it
        if f1_score(data.target[test_idx], preds_formatted, average='macro') > best_score:
            best_ensemble = ensemble

        print('Training {}% complete...'.format(progress/n_splits * 100))
        progress += 1
    
    # Create the description for logging
    log.desc = f'{transf_desc} learning rate: {learning_rate}, n_models: {n_models}, \
    layer shapes: {layer_shapes}, activation: {activation}, subsample: {sample_fraction}, n splits {n_splits}\
    epochs : {epochs}, batch_size : {batch_size}, dropout : {dropout}, label_smoothing : {label_smoothing},  patience : {patience}'

    # Calculate averages of metrics
    log.calculate_statistics()

     # Print results
    log.print_metrics()

    # Save results
    log.save_results()

    return best_ensemble, tfidf

def main(args):
    """
    Used when run from the command line. Creates and trains the model, the saves in in the model_path

    :param args: Parsed args from the command line
    """
    # Set random seed
    tf.random.set_seed(args.seed)
    log = utils.Log(args.results_path, model_name=args.name, desc='')
    model, transformer = neural_network_ensemble(args, preprocess_args=[], log=log)

    # Save the model file
    save_model(args, model, transformer)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)