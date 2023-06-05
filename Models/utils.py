"""
This script provides logging utilities, which log model performance results 
in a single unified dataset.
"""

from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from functools import partial
import os.path
import pandas as pd
import numpy as np
import datetime

class Metric:
    """
    Metric holding class. Accepts a name and a metric function 
    with which to evaluate data. 
    
    This function should have a 
    signature of func(true, pred), where true is a list of true
    target values and pred is the predictions made by the model.

    Holds a list of all metric evaluations and has functions to 
    calculate the means and stddevs of these values.
    """

    def __init__(self, name : str, metric_func):

        self.name = name
        self.metric_func = metric_func
        self.vals = []

    def evaluate(self, true, pred):
        """
        Evaluates the metric.
        """
        self.vals.append(self.metric_func(true, pred))

    def get_mean(self):
        """
        Calculates the mean of the metric evaluation and returns it.
        It will also be available in the .mean attribute.
        """

        self.mean = np.mean(self.vals)
        return self.mean
    
    def get_stddev(self):
        """
        Calculates the standard deviation of the metric evaluation and returns it.
        It will also be available in the .stddev attribute.
        """
        
        self.stddev = np.std(self.vals)
        return self.stddev

class Log:
    """
    Utility class for evaluating and saving metrics. Should be used for one
    experiment and then re-created. Will use the path to load or create a 
    results dataframe, and has functionality to evaluate all added metrics, 
    and will save the metric statistics into this results dataframe.

    Default configuration will track accuracy, f1 macro, f1 weighted and fbeta macro 
    score, with beta=0.5.

    If your metric function has more parameters than metric(true, pred), true
    being true targets and pred being model predictions. Use partial functions 
    to pass the other arguments.

    You can add values to metrics using the add_value() method.
    """

    def __init__(self, path : str, model_name : str, desc :str, default_config : bool = True):
        self.path = path
        self.model_name = model_name
        self.desc = desc

        self.metrics = {}

        # Add default metrics if default_config is true
        if default_config:
            self.default_config()

            # Load results dataframe
            self.load_results()
        
    def add_metric(self, metric_func, name : str):
        """
        Adds a metric to the log.

        metric func should have a signature of 
            metric(true, pred)
        with true being true targets and pred being model predictions.
        Use partial functions to pass the other arguments to the metric func.
        """
        self.metrics[name] = Metric(name, metric_func)

    def add_value(self, val, name : str):
        self.metrics[name].vals.append(val)

    def evaluate_metrics(self, true, pred):
        for metric in self.metrics.values():
            metric.evaluate(true, pred)

    def calculate_statistics(self):
        """
        Calculate statistics for all metrics

        This is currently the mean and standard deviation
        """
        for metric in self.metrics.values():
            metric.get_mean()
            metric.get_stddev()

    def print_metrics(self):
        for metric in self.metrics.values():
            print('Average {} was {:.4f} +- {:.4f}'.format(metric.name, metric.mean, metric.stddev))

    def default_config(self):
        """
        Initialize the Log with default metric configuration.

        This currently consists of accuracy, f1 macro, f1 weighted
        and fbeta macro, beta=0.5
        
        """
        self.add_metric(accuracy_score, 'Accuracy')
        self.add_metric(partial(f1_score, average='macro'), name='F1_macro')
        self.add_metric(partial(f1_score, average='weighted'), name='F1_weighted')
        self.add_metric(partial(fbeta_score, average='macro', beta=0.5), name='F_0.5 macro')
        self.desc = self.desc + ' ' + datetime.datetime.now().strftime('%Y:%m:%d-%H:%M:%S')

    def save_results(self):
        """
        Calculates all metric statistics and saves them to the results dataframe, 
        which will be written to the disk
        """
        # Calculate statistics for every metric
        self.calculate_statistics()
        
        # Add initial data
        res = []
        res.append(self.model_name)
        res.append(self.desc)

        # Add metric means and stddevs
        for metric in self.metrics.values():
            res.append(metric.mean)
            res.append(metric.stddev)

        self.results.loc[len(self.results.index)] = res
        self.results.to_json(self.path, indent=4)
        
    def load_results(self):
        """
        Loads the results dataframe. If it doesn't exist, 
        creates one based on the currently added metrics,
        with columns including their names.
        """
        if os.path.exists(self.path):
            self.results = pd.read_json(self.path)

        else:
            cols = []
            cols.append('Name')
            cols.append('Decsription')
            for metric in self.metrics.values():
                cols.append(metric.name)
                cols.append(f'{metric.name}_stddev')

            self.results = pd.DataFrame(columns=cols)