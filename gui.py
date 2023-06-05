"""
This script provides a GUI for model creation, made using PySimpleGUI.

For usage, run this script from the terminal with the --help command
"""

import PySimpleGUI as sg
import inspect
import threading
import argparse
import sys
import re
from types import FunctionType
from ast import literal_eval as make_tuple
from os.path import dirname, join

# add Models folder to the path, so that file is in the Models package
models_path = join(dirname(__file__), 'Models')
sys.path.append(models_path)

import Models.rnn
import Models.neural_network
import Models.other_models
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default=join('Data', 'corrected_train_dataset_alt.json'), type=str, help="Path to annotated dataset")
parser.add_argument("--results_path", default="results.json", type=str, help="Path to results dataframe")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.2, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument('--n_splits', type=int, default=5, help='Number of splits used for cross-validation')
parser.add_argument("--model_folder", default=".\\Models\\", help="Model folder path")

SPECIAL_ARGS = ['args', 'preprocess_args', 'data', 'log']

def get_model_names(model_funcs : list[tuple[FunctionType, FunctionType, FunctionType]]) -> list[str]:
    """
    Extracts the model names from the first line of their docstrings, and 
    returns them as a list
    :param model_funcs: list of (model_func, prep_func) functions and their preprocess funcs.
    :return: list[str] containing the names of the functions
    """
    names = []
    for func, _, _ in model_funcs:
        for line in func.__doc__.splitlines():
            if line != '':
                names.append(line.strip())
                break

    return names

def thread_train_function(model_func, args, model, transformer):
    """
    Function that is given to the thread, which runs the model training and
    saves the model and transformer in the given model and transformer arguments 
    respectively.

    :param model_func: Function that trains the model, should return a tuple (model, transformer)
    :param args: args for model_func
    :param model: Variable which will hold the resulting model
    :param transformer: Variable which will hold the resulting model
    """
    try:
        model, transformer = model_func(*args)

    except ValueError as e:
        print(f"Incorrect Argument value passed. Full exception: {e}")
    
    except TypeError as e:
        print(f"One or more Arguments have incorrect types. Full exception: {e}")

def extract_arg_description(docstring : str):
    # First capture is the name of the parameter and second is its description
    pattern = re.compile(":param (.+):(.+)")
    res_dict = {} # Dictionary with parameter names as keys and descriptions as values
    for match in re.finditer(pattern, docstring):
        res_dict[match.group(1)] = match.group(2)

    return res_dict

def create_row_with_params(model_func) -> tuple:
    """
    Creates a row for a window based on the function parameters using retrospection

    :param model_func: Function for which to create the row
    :return: tuple (row, signature) containing the created row and the Signature of the function
    """
    args, _, _, defaults, _, _, _ = inspect.getfullargspec(model_func) # Get the arguments and defaults
    row = [] # The row to return
    offset = 0
    if defaults is not None:
        offset = len(args) - len(defaults) # Offset so that the default values can be placed properly
    
    arg_desc = extract_arg_description(model_func.__doc__)
    for i in range(len(args)):
        # Skip special args
        if is_special_arg(args[i]):
            continue
        
        # Every argument will have its name first
        row.append(sg.Text(args[i]))

        # If it has a default value, set it as such
        if i >= len(args) - len(defaults):
            tooltip=None
            if args[i] in arg_desc:
                tooltip = arg_desc[args[i]]
            row.append(sg.Input(default_text=str(defaults[i - offset]), size=(max(12,len(str(defaults[i - offset]))), 1), key=args[i], tooltip=tooltip))
            continue
        
        # Otherwise just make a box
        row.append(sg.Input(size=(12, 1), key=args[i]))

    if row == []:
        row = [sg.Text('No parameters available')]
    return row

def is_special_arg(s : str) -> bool:
    """
    Checks whether a given arguments is special

    :param s: name of the argument
    :return: Bool that's True if the argument is special
    """
    
    return s in SPECIAL_ARGS

def bool_is_true(x : str):
    """
    Whether a string representation of a bool is True or not
    """
    return x.lower() != 'false'

def main(args):
    """
    Main function that handles the GUI creation and responses. Holds the list of model functions
    and creates the GUI layout based on this. Layout changes are handled by closing the old window
    and opening a new one, because PySimpleGUI doesn't support dynamic windows.

    :param args: Arguments parsed from the command line
    """
    # Set a theme for the window
    sg.theme('DarkBlue')

    # List of model functions and their corresponding preprocess functions, and the function to call for training
    # Training function should have a signature train_func(args, model_func, model_args, prep_args)
    model_funcs = [
        (Models.other_models.multi_naive_bayes, Models.other_models.preprocess, Models.other_models.training_wrapper),
        (Models.other_models.complement_naive_bayes, Models.other_models.preprocess, Models.other_models.training_wrapper),
        (Models.other_models.gauss_naive_bayes, Models.other_models.preprocess, Models.other_models.training_wrapper),
        (Models.other_models.svc, Models.other_models.preprocess, Models.other_models.training_wrapper),
        (Models.other_models.random_forest, Models.other_models.preprocess, Models.other_models.training_wrapper),
        (Models.other_models.adaboost, Models.other_models.preprocess, Models.other_models.training_wrapper),
        (Models.other_models.gradient_boosting, Models.other_models.preprocess, Models.other_models.training_wrapper),
        (Models.neural_network.neural_network_ensemble, Models.neural_network.preprocess, Models.neural_network.training_wrapper),
        (Models.rnn.rnn_embed, Models.rnn.preprocess, Models.rnn.training_wrapper)
        ]

    # Extract the names from the docstrings
    model_names = get_model_names(model_funcs)

    # Dict with model names as keys and the previous (model_func, prep_func) tuples as values
    model_lookup_dict = dict(zip(model_names, model_funcs))

    # Intro window layout
    layout = [
        [sg.Push(),sg.Text('Model training interface'), sg.Push()],
        [sg.VPush()],
        [sg.Push(), sg.Text('Model:'), sg.Combo(model_names, size=(40, len(model_names)),
            key = 'MODEL', enable_events=True), sg.Push()],
        [sg.VPush()],
    ]
    
    window = sg.Window('Model Interface', layout, element_justification='l', size=(1200, 800))
    model_sig = object # Model function signature
    prep_sig = object # Preprocess function signature
    while True:
        # Check events
        event, values = window.read()

        # Close the program if the window is closed
        if event == sg.WIN_CLOSED:
            break

        # User chose a model, show the window with hyperparameters and preprocess paramteres
        if event == 'MODEL':
            model_row, model_sig = create_row_with_params(model_lookup_dict[values['MODEL']][0]), inspect.signature(model_lookup_dict[values['MODEL']][0])
            preprocess_row, prep_sig = create_row_with_params(model_lookup_dict[values['MODEL']][1]), inspect.signature(model_lookup_dict[values['MODEL']][1])
            
            # List of rows that are the result of splitting the original one
            model_rows = []
            for i in range(0,len(model_row), 8):
                if len(model_row) - i <= 8:
                    # Not enough elements for a full row, break
                    model_rows.append(model_row[i:])
                    break

                model_rows.append(model_row[i : i + 8])

            # List of rows that are the result of splitting the original one
            preprocess_rows = []
            for i in range(0, len(preprocess_row), 8):
                if len(model_row) - i <= 8:
                    # Not enough elements for a full row, break
                    preprocess_rows.append(preprocess_row[i:])
                    break

                preprocess_rows.append(preprocess_row[i : i + 8])
            
            new_layout = [
                [sg.Push(),sg.Text('Model training interface'), sg.Push()],
                [sg.VPush()],
                [sg.Push(), sg.Text('Model:'), sg.Combo(model_names, default_value=values['MODEL'], 
                    size=(40, len(model_names)), key = 'MODEL', enable_events=True), sg.Push()],
                [sg.VPush()],
                [sg.Text('Global Parameters:')],
                [sg.Text('Random Seed:'), sg.Input(default_text='42', size=(5, 1), key='seed', enable_events=True),
                    sg.Text('Test Size:'), sg.Input(default_text='0.2', size=(5, 1), key='test_size', enable_events=True),
                    sg.Text('Force Dense:'), sg.Input(default_text='False', size=(5, 1), key='force_dense', enable_events=True),
                    sg.Text('Save model:'), sg.Input(default_text='True', size=(5, 1), key='save', enable_events=True)],
                [sg.VPush()],
                [sg.Text('Model parameters:')],
                *model_rows,
                [sg.VPush()],
                [sg.Text('Preprocess parameters:')],
                *preprocess_rows,
                [sg.VPush()],
                [sg.Push(), sg.Button('Train', enable_events=True, key='TRAIN'), sg.Push()],
                [sg.Push(), sg.Multiline(size=(200, 10), autoscroll=True, reroute_stderr=True, reroute_stdout=True, key='OUTPUT'), sg.Push()],
                [sg.VPush()]
            ]

            # Close the old window
            window.close()

            # Create a new window with the updated layout
            window = sg.Window('Model Interface', new_layout, element_justification='l')

        # User chose to train the model
        if event == 'TRAIN':
            try:
                args.seed = int(values['seed'])
                args.test_size = float(values['test_size'])
                args.force_dense = bool_is_true(values['force_dense'])
                args.save = bool_is_true(values['save'])
                model_param_package = [] # list of arguments for the model function
                prep_param_package = [] # list of arguments for the preprocess function

                # Convert the arguments from the string values in the window to their corresponding types from the function signatures
                for params, package in [(model_sig.parameters, model_param_package), (prep_sig.parameters, prep_param_package)]:
                    for param in params:

                        # If it's a special kind of argument, skip it
                        if is_special_arg(param):
                            continue
                        
                        # If the argument has a default value, the type is its type
                        if params[param].default is not inspect.Parameter.empty:
                            t = type(params[param].default)
                            if t is tuple:
                                package.append(make_tuple(values[param]))
                                continue
                            
                            if t is bool:
                                package.append(bool_is_true(values[param]))
                                continue

                            package.append(t(values[param]))

                        # Otherwise the function will have an annotation for this argument,
                        # if not, throw an exception
                        else:
                            t = params[param].annotation

                            if t is inspect.Parameter.empty:
                                raise TypeError('Function argument has no type annotation')

                            # If it's a tuple, it has to be parsed correctly
                            if t is tuple:
                                package.append(make_tuple(values[param]))
                                continue

                            package.append(t(values[param]))
                
                # Train the model in a different thread
                model = object
                transformer = object

                thread = threading.Thread(target=thread_train_function, args=
                (
                    model_lookup_dict[values['MODEL']][2],
                    (args, model_lookup_dict[values['MODEL']][0], model_param_package, prep_param_package),
                    model,
                    transformer
                ),
                daemon=True)
                thread.start()

                print('Model training started.')

            except ValueError:
                print('Couldn\'t start model training. One or more parameters have invalid value format')

    window.close()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

