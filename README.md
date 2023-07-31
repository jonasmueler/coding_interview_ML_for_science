### Coding Interview

## Neural Network Training with Random Search Hyperparameter Optimization for Tabular Data

This github repository is designed to automatically derive an optimized neural network architecture and hyperparameters with random search hyperparameter optimization. 
The Full code is controlled via a bash script control_script.sh which gets data from the openml library, preprocesses and splits this data (dataset_preprocessing.py), trains an XGBoost classifier as a comparison baseline (XGBoost.py), derives the neural network with random search optimization (neural_net_random_search.py), visualizes the parameter and performance trajectory of the best hyperparameter combination (MLP_visualizations.py) and tests the two models on the testset (performance_test.py). 

The Tool is designed to be used from the command line. All outputs and error messages are logged in text files in the /logs directory. Models are saved in /models, plots in /plots and the used data split in /data.




