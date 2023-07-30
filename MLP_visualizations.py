import numpy as np
from config import *

##
import matplotlib.pyplot as plt # imported for plotting 
import os # for directory use

def plot_data()-> None:
    # get data 
    os.chdir(os.path.join(path_origin, "optimization_results_MLP"))
    best_pars = np.load(os.path.join(path_origin, "optimization_results_MLP", "best_parameter_combination_over_time.npy"))
    
    # Names for the parameters
    param_names = ["Learning Rate", "Weight decay", "Batch Size", "Layer Size", "Number Layers", "F1 Score"]

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True)

    # Plot each parameter in a subplot
    for i, ax in enumerate(axs.flatten()):
        param_name = param_names[i]
        ax.plot(best_pars[:, i], label=param_name)
        ax.set_title(param_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')

    # Adjust layout
    plt.tight_layout()

    # save plot 
    os.chdir(path_origin)
    path = os.path.join(path_origin, "plots")
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    plt.savefig('parameter_trajectory_random_search_MLP.pdf', dpi = 600)
    os.chdir(path_origin)
    
    # Show the plot
    #plt.show()

    return


if __name__ == "__main__":
    print("##################################################################################################")
    plot_data()
