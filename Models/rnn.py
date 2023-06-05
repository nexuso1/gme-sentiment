"""
This script handles the training of RNN models.

For usage, run this script from the terminal with the --help command
"""

import tensorflow as tf
import tensorflow_addons as tfa
import tensorboard
import numpy as np
import re
import pandas as pd
import argparse
import utils
import os
import datetime

from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default= os.path.join('Data', 'corrected_train_dataset_alt.json'), type=str, help="Path to annotated dataset")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.2, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument('--cell', type=str, default='lstm', help='Recurrent cell type')
parser.add_argument('--n_splits', type=int, default=5, help='Number of splits used for cross-validation')
parser.add_argument('--name', type=str, default='rnn', help='Custom model name')
parser.add_argument('--desc', default='LSTM', help='Model description in the results log')
parser.add_argument('--save', type=bool, default=True, help="Whether to save the trained model")
parser.add_argument("--model_folder", default="Models", help="Model folder path")
parser.add_argument("--results_path", default="results_new.json", type=str, help="Path to results dataframe")


def preprocess(data : np.ndarray, shorten=True, max_length='auto', window=10, keywords=['gme', 'gamestop'], sep='...'):
    """
    Text Preprocessor 
 
    Preprocess the text, replacing links with '[link]' and other
    special character removals. Can also shorten the text.

    :param data: Source Dataframe
    :param shorten: Whether to use text shortening or not
    :param window: Window size for text shortening
    :param max_length: Maximum length in tokens of the shortened text. Either 'auto' or an int. 'auto' sets the maximum length to the 60th quantile of lengths. Also, the longest example is removed before the quantile calculation and text shortening
    :param keywords: List of strings, which are used as the keywords in the shortening process
    :param sep: String placed at the beginning and end of every non-overlapping window
    :return: Preprocessed text
    """

    if shorten:
        lengths = np.array(list(map(len, data)))
        np.delete(data, np.argmax(lengths), axis=0)

        if max_length == 'auto':
            max_length = np.quantile(lengths, 0.6)

        # Initial text cleanup
        for i in range(data.shape[0]):
            post = data[i]
            data[i] = re.sub('http[\w:/\.\=\#\?\-\$\&]+', '[link]', post)

            if len(post) >= max_length:
                data[i] = shorten_text(data[i], window=window, keywords=keywords, sep=sep, max_length=max_length)
    return data

def build_vocab(data, mode : str, max_tokens : None|int):
    """
    Creates a vocabulary out of the data. Tokens can be words or characters,
    denoted by the mode variable. Maximum vocabulary size can also be set.
    Returns a list of the tokens in the vocabulary, in descending order by frequency.

    :param data: Source data, iterable made out of strings
    :param mode: Either 'character' for character tokens, or 'word' for word tokens
    :return: list of the tokens in the vocabulary, in descending order by frequency
    """
    freqs = {}
    for string in data:
        if mode == 'character':
            iterated = string
        else:
            iterated = string.split()
        for char in iterated:
            if char not in freqs:
                freqs[char] = 0

            freqs[char] += 1

    vocab = sorted(freqs.keys(), key=lambda x: freqs[x], reverse=True)
        
    if max_tokens is not None:
        vocab = vocab[:max_tokens]

    return list(vocab)

def compile_model(model : tf.keras.Model, n_classes, learning_rate, label_smoothing):
    """
    Compiles the given model with the input hyperparameters

    :param model: Model to be compiled
    :param n_classes: Number of output classes
    :param learning_rate: Learning rate to be used during training
    :param label_smoothing: Strength of label smoothing to use during traning (set to 0 if not necessary)
    """

    # Set metrics for accuracy
    metrics= [
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'), 
        tfa.metrics.F1Score(n_classes, average='weighted', name='f1_weighted'),
        tfa.metrics.F1Score(n_classes, average='macro', name='f1_macro'),
        tfa.metrics.FBetaScore(n_classes, average='macro', beta=0.5, name='f_0.5 macro')
        ]
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=metrics,
    )

    return model

def make_rnn_embed(n_words, layers_shape, rnn_cell, n_classes = 3):
    """
    Creates the RNN model with a given architecture

    :param n_words: Number of words in the vocabulary/alphabet (used for emebeddings)
    :param n_classes: Number of classes to predict
    :param layers_shape: Shape of the embedding and RNN layers, with the first
    being the output size of the embedding, and the rest output sizes of the rnn layers
    :param rnn_cell: RNN cell type ('lstm' or 'gru')
    :param learning_rate: Learning rate used during training

    :return: Compiled RNN model
    """
    # Variable length input
    text_input = tf.keras.Input(shape=[None], ragged=True, name='input')
    
    # Embed the words into vectors of uniform length
    embedding = tf.keras.layers.Embedding(n_words, output_dim=layers_shape[0])(text_input)
    last_layer = embedding

    for i in range(1, len(layers_shape)):
        # Run the sequnece through the RNN layer
        ret_seq = True
        if i == len(layers_shape) - 1:
            ret_seq = False

        if rnn_cell == 'lstm':
            new_layer = tf.keras.layers.LSTM(layers_shape[i], return_sequences=ret_seq)(last_layer)

        elif rnn_cell == 'gru':
            new_layer = tf.keras.layers.GRU(layers_shape[i], return_sequences=ret_seq)(last_layer)

        last_layer = new_layer

    # Predict the resulting classes with a softmax
    classes_pred = tf.keras.layers.Dense(n_classes, activation='softmax')(last_layer)

    model = tf.keras.Model(inputs=text_input, outputs=classes_pred)
    print(model.summary())
    
    return model

def shorten_text(text : str, window : int, keywords=['gme', 'gamestop'], sep='...', max_length=None):
    '''
    Shortens the text to windows of size at least 'window * 2 + 1'.
    These windwos contain words around the occurences of words in the list
    'keywords'. If windows should overlap, then they will be merged together 
    in the output string. Windows are separated by the string 'sep'.

    :param window: Window size for text shortening
    :param max_length: Maximum length in tokens of the shortened text. Either 'auto' or an int. 'auto' sets the maximum length to the 60th quantile of lengths. Also, the longest example is removed before the quantile calculation and text shortening
    :param keywords: List of strings, which are used as the keywords in the shortening process
    :param sep: String placed at the beginning and end of every non-overlapping window
    '''
    res = []
    split_text = text.split()
    new_sequence = True
    pattern = '|'.join(keywords)
    to_add = 0
    for i in range(len(split_text)):
        if max_length is not None and len(res) > max_length:
            res.append(sep)
            break

        word = split_text[i]
        if to_add > 0:
            res.append(word)
            to_add -= 1

            if to_add == 0:
                res.append(sep)
                new_sequence = True

        if re.match(pattern, word, re.IGNORECASE):
            if new_sequence:
                res.append(sep)
                for j in range(max(0, i - window), i - 1):
                    res.append(split_text[j])

                res.append(word)
                new_sequence = False

            to_add = window + 1
    
    return ' '.join(res)

def create_datasets(args, features, target, train_idx, test_idx, batch_size, standardize, split, max_vocab_size):
    """
    Creates the datasets for training and splits them into batches.
    Also vectorizes them using a TextVectorization layer.

    :param args: Command Line arguments
    :param train_idx: List of indices to be used for training
    :param test_idx: List of indices to be used for testing
    :param batch_size: Batch size for training
    :param standardize: Type of standardization to use. Available choices are 'lower', 'strip_punctuation', 'lower_and_strip_punctuation'
    :param split: Type of string splitting to use. Available choices are 'whitespace', 'character'
    :param max_vocab_size: Maximum vocabulary size to use when creating tokens
    :return: Tuple of train_data, test_data, val_data, text_vectorizer,
    containing all the training data in batches, test data and so on, and
    a vectorizer layer that was adapted to the training data
    """

    # Stratify split the data into train, test and validation sets
    target = tf.keras.utils.to_categorical(target)
    train_data, train_target = features[train_idx], target[train_idx]
    test_data, test_target = features[test_idx], target[test_idx]

    # Create vectorizer layer
    text_vectorizer = tf.keras.layers.TextVectorization(standardize=standardize, max_tokens=max_vocab_size, split=split, ragged=True)
    
    # # Build vocabulary (has to be done since we cannot save in a file the vocabulary
    # # generated by the text_vectorizer, because of some internal encoding error)
    # vocab = build_vocab(train_data, split, max_vocab_size)
    # text_vectorizer.set_vocabulary(vocab)

    text_vectorizer.adapt(train_data)

    # Transform text
    train_data = text_vectorizer(train_data) 
    test_data = text_vectorizer(test_data)

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_target)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_target)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size))

    train_ds = train_ds.shuffle(len(train_ds), args.seed)

    return train_ds, test_ds, text_vectorizer

def train_model(train_ds, val_ds, model : tf.keras.Model, logdir, epochs=20, patience=5):
    """
    Function that handles model training. Can customize the amount of epochs and patience for early stopping.

    :param train_ds: Train dataset
    :param val_ds: Validation dataset
    :param model: compiled Model instance
    :param epochs: Number of training epochs
    :param patience: Patience for early stopping
    :return: Trained model
    """
    # Set early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        restore_best_weights=False
    )

    tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop, tb_callback])

    return model

def training_wrapper(args, model_func, model_args, prep_args):
    """
    Training wrapper for interaction with GUI.
    Trains and saves the model, and returns a (model, transformer) tuple

    :param args: Command line args
    :param model_func: Function that will handle the training
    :param model_args: Arguments for the model training function
    :param prep_args: Arguments for the preprocess function
    :return: A tuple containing the model and transformer
    """
    # Create log
    log = utils.Log(args.results_path, model_name='', desc='')
    # Train model
    model, transformer = model_func(args, log, prep_args, *model_args)

    if args.save:
        # Save the model
        save_model(args, 'rnn', model, transformer)

    return model, transformer

def rnn_embed(args, log : utils.Log, preprocess_args : tuple = [], cell_type = 'lstm', layers_shape=(32, 64, 64), 
        epochs=50, batch_size=32, learning_rate=0.001, label_smoothing=0.1, standardize = 'lower_and_strip_punctuation',
        max_vocab_size=7000, split='whitespace',  patience=20):
    """
    RNN with Word embeddings
    
    Handles the creation and training of the model

    :param layers_shape: Shape of the embedding and RNN layers, with the first
    being the output size of the embedding, and the rest output sizes of the rnn layers
    :param cell_type: RNN cell type ('lstm' or 'gru')
    :param batch_size: Batch size during training
    :parma preprocess_args: Parameters for preprocessing
    :param learning_rate: Learning rate used during training
    :param patience: Patience for early stopping
    :param label_smoothing: Strength of label smoothing to use during traning (set to 0 if not desired)
    :param standardize: Type of standardization to use. Available choices are 'lower', 'strip_punctuation', 'lower_and_strip_punctuation'
    :param split: Type of string splitting to use. Available choices are 'whitespace', 'character'
    :param max_vocab_size: Maximum vocabulary size to use when creating tokens

    :return: Trained model and text vectorizer
    """
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(args.seed)

    # Load data
    data = pd.read_json(args.dataset_path, encoding='utf-8')
    target = data.target.to_numpy()

    # Update model name and description in the log

    log.desc = f'embedding size: {layers_shape[0]}, cell output size: {layers_shape[1]}, epochs: {epochs}, batch_size: {batch_size}, learning rate{learning_rate}, patience: {patience}'
    progress = 1
    for train_idx, test_idx in StratifiedKFold(n_splits=args.n_splits).split(data.text, target):
        # Preprocess data
        features = preprocess(data.text.to_numpy(), *preprocess_args)
        # Split data in the fold to train test and validation sets
        train_ds, test_ds, vectorizer = create_datasets(args, features, target, train_idx, test_idx, batch_size=batch_size,
                                                                standardize=standardize, max_vocab_size=max_vocab_size,
                                                                 split=split)

        logdir = os.path.join('Logs', cell_type + '-' + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))

        # Train the model
        model = make_rnn_embed(vectorizer.vocabulary_size(), layers_shape, cell_type, n_classes = 3)
        model = compile_model(model, 3, learning_rate, label_smoothing = label_smoothing)
        model = train_model(train_ds, test_ds, model, logdir, epochs, patience)
        
        # Calculate metrics
        loss, accuracy, f1_macro, f1_weighted, f05_macro = model.evaluate(test_ds)
        
        log.add_value(accuracy, 'Accuracy')
        log.add_value(f1_macro, 'F1_macro')
        log.add_value(f1_weighted, 'F1_weighted')
        log.add_value(f05_macro, 'F_0.5 macro')

        print('Training {}% complete...'.format(progress/args.n_splits * 100))
        progress += 1

    log.model_name = cell_type

    # Calculate metric statistics
    log.calculate_statistics()

    # Print metric info
    log.print_metrics()

    # Save experiment results
    log.save_results()

    return model, vectorizer

def save_model(args, name : str, model : tf.keras.Model, vectorizer):
    """
    Saves the model and transfromer in a file, with a name
    specified in args.name, or cell type otherwise.

    :param name: Custom model name
    :param model: Model to be saved
    :param vectorizer: A vectorizer instance to be saved
    """

    if not os.path.exists(f'{args.model_folder}/{name}.MODEL/'):
        os.mkdir(f'{args.model_folder}/{name}.MODEL')
    model.save(f'{args.model_folder}/{name}.MODEL')

    if not os.path.exists(f'{args.model_folder}/{name}_vectorizer.MODEL/'):
        os.mkdir(f'{args.model_folder}/{name}_vectorizer.MODEL/')

    vec_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, ), dtype=tf.string),
        vectorizer
    ])
    vec_model.save(f'{args.model_folder}/{name}_vectorizer.MODEL/')
    print('Model saved.')

def main(args):
    # Determine model name
    if args.name is not None:
        name = args.name

    else:
        name = args.cell

    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(args.seed)

    # Create log
    log = utils.Log(args.results_path, name, desc=args.desc)

    # Train model
    model, vectorizer = rnn_embed(args, log=log)

    save_model(args, name, model, vectorizer)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)