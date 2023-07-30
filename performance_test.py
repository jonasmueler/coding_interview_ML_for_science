from xgboost import XGBClassifier
import sklearn
import numpy as np
from config import *
from neural_net_random_search import MLP
from XGBoost import get_data
import torch

# 
import os # paths


def testset_performance()-> None:
    """
    Loads the trained Neural Network and XGBoost models and test their performance on the testset
    """ 
    print("Start testing")
    # load models
    os.chdir(os.path.join(path_origin, "models"))
    ## XGBoost
    XGBoost = XGBClassifier()
    XGBoost.load_model("XGBoost.model")

    ## neural network
    # get optimized parameter set and architecture 
    os.chdir(os.path.join(path_origin, "optimization_results_MLP"))
    best_pars = np.load(os.path.join(path_origin, "optimization_results_MLP", "best_parameter_combination_over_time.npy"))
    layers = int(best_pars[-1, 4])
    layer_size = int(best_pars[-1, 3])

    # init model
    path = os.path.join(path_origin, "models", "optimizd_MLP.pth")
    NN = MLP(20, 1, layer_size, layers).float()

    # load weights
    NN.load_state_dict(torch.load(path))
    NN.eval()
    print("Models loaded")

    # get test data
    _, _, X_test, Y_test = get_data()
    print("Data loaded")

    # get predictions 
    MLP_pred = NN(torch.from_numpy(X_test)).detach().numpy()
    MLP_pred[MLP_pred > 0.5] = 1
    MLP_pred[MLP_pred < 0.5] = 0
    XGBoost_pred = XGBoost.predict(X_test)
    
    # F1
    f1 = sklearn.metrics.f1_score
    MLP_score = f1(MLP_pred, Y_test)
    XGBoost_score = f1(XGBoost_pred, Y_test)

    print(f"F1 score MLP: {MLP_score}, F1 score XGBoost: {XGBoost_score}")

    return  

if __name__ == "__main__":
    testset_performance()


