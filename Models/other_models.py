"""
This script handles the training of all sklearn models used.

For usage, run this script from the terminal with the --help command
"""

import re
import os
import sklearn
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.feature_extraction.text
import sklearn.tree
import sklearn.svm
import pandas as pd
import pickle
import lzma
import argparse
from utils import Log
import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default=".\\Data\\corrected_train_dataset_alt.json", type=str, help="Path to annotated dataset")
parser.add_argument("--results_path", default="results_new.json", type=str, help="Path to results dataframe")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.2, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--model_folder", default="Models", help="Model folder path")
parser.add_argument("--save", default=True, type=bool, help='Whether or not to save the trained model in the Models folder.')
parser.add_argument("--force_dense", default=True, type=bool, help="Force dense representation of the feature matrix.")

def to_dense(x):
    """
    Helper function for preprocessing in a Pipeline. Turns the
    sparse matrix given to this function into a dense matrix

    :param x: Sparse matrix
    :return: Dense form of matrix x 
    """
    return x.toarray()

def preprocess(data : pd.DataFrame, min_df=5, ngram_range=(1,2), use_pca=True, replace_links=False, n_components=200, analyzer='word'):
    """
    Preprocess the text, replacing links with '[link]' and other
    special character removals, and transform it via tfidf vectorizer.
 
    :param data: Source Dataframe
    :param min_df: Minimum df for tfidf
    :param ngram_range: ngram size to consider when creating features
    :param analyzer: Whether to take into account word ngrams or character ngrams. Values can be 'word' or 'char'
    :param use_pca: Whether to use PCA or not
    :param replace_links: Whether to replace links in the text or not
    :param n_components: Number of principal components to use in PCA

    :return: (Preprocessed text, tfidf vectorizer, vectorizer description)
    """

    # Replace links first
    if replace_links:
        for i in data.index:
            post = data.loc[i]
            # post = re.sub(pattern='\n', repl=' ', string=post) # remove newline chars
            data.at[i] = re.sub(pattern='http[s]*:\/\/[^\s]+', repl='[link]', string=post)

    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, analyzer=analyzer)
    features = tfidf.fit_transform(data)
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

    return features, pipe, desc

def svc(args, C=10/3):
    """
    SVC

    Support vector machine classifier.

    :param C: Regularization paramter to use during training
    :return: SVC instance
    """
    return sklearn.svm.SVC(C=C, cache_size=2000, random_state=args.seed, verbose=False, tol=0.0001)

def gradient_boosting(args, n_estimators = 200, min_samples_leaf = 2, learning_rate = 0.5, criterion = 'friedman_mse', max_features=0.3):
    """
    Gradient Boosting Decision Trees

    Model that uses the gradient boosting method to train an ensemble of decision trees.

    :param args: input args, mostly needed for a seed
    :param n_estimators: Number of estimators in the ensemble
    :param min_samples_leaf: minimum number of samples a node must have to be a leaf
    :param criterion: What criterion to use for the calculation, 'friedman_mse' or 'squared_error' available
    :return: Gradient Boosting Classifier instance
    """
    return sklearn.ensemble.GradientBoostingClassifier(
        n_estimators=n_estimators, 
        min_samples_leaf=min_samples_leaf, 
        random_state=args.seed, 
        learning_rate=learning_rate,
        max_features=max_features,
        criterion=criterion,
        verbose=0)

def adaboost(args, n_estimators=500, min_samples_leaf=2, learning_rate=0.1, criterion = 'gini', max_features=0.3):
    """
    Adaboost with Decision Trees

    Model that uses the Adaboost method with a Decision Tree classifier as the base.

    :param args: input args, mostly needed for a seed
    :param n_estimators: Number of estimators in the ensemble
    :param min_samples_leaf: minimum number of samples a node must have to be a leaf
    :param criterion: What criterion to use for the calculation, 'gini' or 'entropy' available
    :param max_features: Naximum number of features to consider during splitting
    :return: Adaboost classifier instance
    """
    return sklearn.ensemble.AdaBoostClassifier(
        base_estimator=sklearn.tree.DecisionTreeClassifier(criterion=criterion, 
            min_samples_leaf=min_samples_leaf, max_features=max_features), 
            n_estimators=n_estimators, 
            random_state=args.seed, 
            learning_rate=learning_rate)

# Create a random forest model
def random_forest(args, n_estimators = 500, min_samples_leaf=2, criterion='gini', max_features=0.3, ccp_alpha=0):
    """
    Random Forest Ensemble

    :param args: input args, mostly needed for a seed
    :param n_estimators: Number of estimators in the ensemble
    :param min_samples_leaf: minimum number of samples a node must have to be a leaf
    :param criterion: What criterion to use for the calculation, 'gini' or 'entropy' available
    :param max_features: Naximum number of features to consider during splitting
    :return: Random forest classifier instance
    """
    return sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion=criterion, 
        min_samples_leaf=min_samples_leaf, 
        n_jobs=-1, 
        random_state=args.seed,
        max_features=max_features,
        ccp_alpha=ccp_alpha,
        verbose=0)

def complement_naive_bayes(args, alpha=0.1):
    """
    Complement Mulitnomial Naive Bayes

    :param args: not needed, just for compatibility
    :param var_smoothing: Laplace smoothing
    :return: Multinomial NB instance with the given alpha
    """
    return sklearn.naive_bayes.ComplementNB(alpha=alpha)

def gauss_naive_bayes(args, var_smoothing=4e-4):
    """
    Gaussian Naive Bayes

    :param args: not needed, just for compatibility
    :param var_smoothing: Laplace smoothing
    :return: Gaussian NB instance with the given smoothing
    """
    return sklearn.naive_bayes.GaussianNB(var_smoothing=var_smoothing)

# Create a NB model
def multi_naive_bayes(args = None, alpha=1):
    """
    Multiomial Naive Bayes

    :param args: not needed, just for compatibility
    :param alpha: Laplace smoothing
    :return: Multinomial NB instance with the given alpha
    """
    return sklearn.naive_bayes.MultinomialNB(alpha=alpha)

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

    # Load the dataset
    df = pd.read_json(args.dataset_path)
    
    # Create a log
    log = utils.Log(path=args.results_path, model_name='', desc='text_cleanup')

    model, transformer = train_model(args, df, model_func(args, *model_args), log, prep_args)
    if args.save:
        with lzma.open(os.path.join(args.model_folder, model_func.__name__) + '.MODEL', 'w') as file:
                pickle.dump((model, transformer), file)
    
        print('Model saved to ' + os.path.join(args.model_folder, model_func.__name__) + '.MODEL')

    return model, transformer

def train_model(args, data : pd.DataFrame, model, log : utils.Log, preprocess_args : list|tuple, n_splits = 5):
    """
    Main model training function.
    Takes the given model and data, preprocesses the data and fits the model to it. Also performs stratified K-fold 
    cross-validation, with K = n_splits.
    :param args: Command line arguments
    :param model: Sklearn model instance
    :param log: Log instance, used for model performance logging
    :param preprocess_args: arguments for the preprocess function
    :param n_splits: Number of folds to use in the K-fold cross-validation
    """
    
    # Target can be taken as is, sklearn models will encode it themselves
    target = data.target

    # Use Stratified K-fold cross-validation to train the model
    stf = StratifiedKFold(n_splits = n_splits, random_state=args.seed, shuffle=True)
    progress = 1

    print('Training started...')
    for train_idx, test_idx in stf.split(data, target):
        # Split the data into training and testing set
        train_data, test_data = data.text[train_idx], data.text[test_idx]
        train_target, test_target = target[train_idx], target[test_idx]

        # Preprocess input data
        train_data, transformer, transf_desc = preprocess(train_data, *preprocess_args)

        if args.force_dense and not isinstance(train_data, np.ndarray):
            train_data = train_data.toarray()
        # Train the model on training data
        model.fit(train_data, train_target)

        # Make predictions for test data
        test_data = transformer.transform(test_data)
        if args.force_dense and not isinstance(test_data, np.ndarray):
            test_data = test_data.toarray()

        preds = model.predict(test_data)

        # Calculate metrics
        log.evaluate_metrics(test_target, preds)

        print('Training {}% complete...'.format(progress/n_splits * 100))
        progress += 1

    # Add model name and transformer description to log
    log.model_name = str(model)
    log.desc += f' {transf_desc}'

    # Calculate metric statistics
    log.calculate_statistics()

    # Print results
    log.print_metrics()

    # Save the results
    log.save_results()

    return model, transformer

def main(args):
    # Load the dataset
    df = pd.read_json(args.dataset_path)

    # run_tests(args, df, results )
    # Choose the model
    model_func = gauss_naive_bayes

    # Model parameters description to be saved in the df
    description = "cleaned"

    # Create a log with the descriptions
    log = Log(args.results_path, model_name="", desc=description)
    
    # Train it and save the model, transformer pair
    model = train_model(args, df, model_func(args), log, [])

    # Save the model and transformer in a file
    with lzma.open(os.path.join(args.model_folder, model_func.__name__) + '.MODEL', "wb") as model_file:
            pickle.dump(model, model_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)