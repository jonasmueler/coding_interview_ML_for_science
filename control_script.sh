#!/bin/bash

# Set directory
log_directory="/home/jonas/PhD_Ml_science/coding_interview_ML_for_science/logs"

# Find the latest run number through checking existing log files
run_number=$(ls "$log_directory" | grep -oE '[0-9]+' | sort -rn | head -1)
((run_number++))  # Increment 

# Set the log filename with run number
log_filename="$log_directory/log_${run_number}.txt"

cd /home/jonas/PhD_Ml_science/coding_interview_ML_for_science

# Function to run Python script and log
run_python_script() {
    python_script="$1"
    python3 "$python_script" >> "$log_filename" 2>&1
}

# run python scripts
run_python_script "dataset_preprocessing.py"
run_python_script "XGBoost.py"
run_python_script "neural_net_random_search.py"
run_python_script "MLP_visualizations.py"
run_python_script "performance_test.py"

