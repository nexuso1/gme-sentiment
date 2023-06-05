# Ročníkový projekt a bakalárska práca Samuela Fančiho

Ročníkový projekt Samuela Fančiho (NPRG045)

## Folders
- ``Data`` - Contains various generated and annotated datasets
- ``Graphs`` - Contains visualizations of the data and results
- ``Logs`` - Contains model training logs 
- ``Models`` - Contains model training scripts and the best model
- ``Networks`` - Contains networks created from the Twitter and Enron data, 
  together with metrics for each node. Networks are saved in edgelist format.

## Scripts
- ``annotator.py`` - Script for manual annotation of posts
- ``scraper.py`` - Script for downloading relevant data from the web
- ``dataset_processor.py`` - Script that transforms data stored in .json files created by ``annotator.py`` and ``scraper.py`` into pandas DataFrame,
                         and also saves it in .json format
- ``Models/rnn.py`` - Script for training RNN models
- ``Models/neural_network.py`` - Script for training neural networks
- ``Models/other_models.py`` - Script for training all other models
- ``network_metrics.py`` - Script that calculates relevant network metrics
- ``data_visualization.ipynb`` - Notebook used for some data visualization
- ``enron_network.py`` - Script used to create the enron network out of emails
- ``experiments.ipynb`` - Notebook used to run some experiments for sentiment models
- ``gamma_estimate.r`` - R script for calculation of the scaling parameter, and goodness-of-fit test for the power-law
- ``gui.py`` - A graphical interface for training models
- ``postprocessing.py`` - Script for resolving Twitter IDs to usernames, and some other helper functions
- ``sentiment.py`` - Script that uses a model to estimate the sentiment of a given dataset, also plots the result as a bar plot
- ``twitter_network.py`` - Script used for the creation of the twitter network
