#!/bin/bash

# Set the log directory
log_directory="/home/jonas/PhD_Ml_science/coding_interview_ML_for_science/logs"

# Find the latest run number by checking existing log files
run_number=$(ls "$log_directory" | grep -oE '[0-9]+' | sort -rn | head -1)
((run_number++))  # Increment the run number for the next run

# Set the log filename with the run number
log_filename="$log_directory/log_${run_number}.txt"

cd /home/jonas/PhD_Ml_science/coding_interview_ML_for_science

# preprocessing
python3 dataset_preprocessing.py >> "$log_filename"

# train XGBoost
python3 XGBoost.py >> "$log_filename"

# train neural net with random search optimization of hyperparameters
python3 neural_net_random_search.py >> "$log_filename"

# visualize results of random search
python3 MLP_visualizations.py >> "$log_filename"

# check testset performance
python3 performance_test.py >> "$log_filename"
