from xgboost import XGBClassifier
import sklearn
import numpy as np
from config import *

##
import os

def get_data()-> np.ndarray:
    """
    Loads data from stored files in data folder
    """
    path = path_origin
    os.chdir(os.path.join(path, "data"))

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    return X_train, y_train, X_test, y_test


def main():
    """
    high level wrapper function
    """

    print("Start Training XGBoost classifier")
    X_train, y_train, X_test, y_test = get_data() 

    # get model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # save trained model
    path = path_origin
    os.makedirs(os.path.join(path, "models"), exist_ok=True)
    os.chdir(os.path.join(path, "models"))
    model.save_model('XGBoost.model')
    os.chdir(path_origin)
    print("Model saved")

if __name__ == "__main__":
    print("##################################################################################################")
    main()

















#if __name__ == "__main__":

