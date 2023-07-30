import openml
import sklearn
import numpy as np
from config import *

## imported packages
import pandas as pd # for signature of functions to use output of openml.datasets.get_dataset(), otherwise only numpy used
import os # store files in directory

def get_dataset(did: int = 31) -> tuple[np.ndarray, ...]:
    """Download dataset from OpenML. Dataset id 31 is credit-g and 43898 is adult
    (for more details see openml.org/d/<did>). Both are fine for this task."""
    dataset = openml.datasets.get_dataset(did)
    X, y, cats, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    _y = sklearn.preprocessing.LabelEncoder().fit_transform(y).reshape([-1, 1])
    return X, _y, cats

def check_missings(X: pd.DataFrame) -> pd.DataFrame:
    """
    checks the amount of missing values in the data ,sums over columns in output
    """
    return X.isnull().sum(axis = 0)

def preprocess_data(X: pd.DataFrame, y: np.ndarray, cats: list, standardize: bool) -> list[np.ndarray, np.ndarray]:
    """
    Takes the features as X, converts categorical data to floats and standardizes the other numerical variables
    returns: list[features, targets]
    """

    # get categorical variables in array and transform
    X = np.array(X)
    X_shape = X.shape
    categoricals =  X[:, cats]
    encoder = sklearn.preprocessing.OrdinalEncoder()
    encoded_features = np.array(encoder.fit_transform(categoricals))

    # put back into original array
    X[:, cats] = encoded_features

    # check 
    assert X.shape == X_shape
    
    
    # transform numerical variables
    numericals = X[:, np.invert(cats)]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(numericals)
    numericals = scaler.transform(numericals)
    X[:, np.invert(cats)] = numericals
    
    # check 
    assert X.shape == X_shape

    # change datatype
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return [X, y]


def train_test_split(X: np.ndarray, y: np.ndarray, split_criterion: float) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    #Takes data and splits into train and test data based on split_criterion 
    """

    # create train and test splits
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=split_criterion, random_state=42)

    # save on hard-drive 
    path = path_origin
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    os.chdir(os.path.join(path, "data"))
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
    np.savetxt("X_train.csv", X_train, delimiter=",")
    np.savetxt("X_test.csv", X_test, delimiter=",") 
    np.savetxt("y_train.csv", y_train, delimiter=",") 
    np.savetxt("y_test.csv", y_test, delimiter=",") 
    os.chdir(path_origin) 
    print("Train and test data saved")
   
    return 

def main(check_missings: bool):
    """
    high level wrapper function
    """
    dataset = get_dataset() ## features (1000, 20), targets (1000, 1), features (pandas dataframe), targets (numpy ndarray)
    print("Start preprocessing")
    print("Data loaded")

    ## check missings
    if check_missings: 
        print(check_missings(dataset[0]))

    ## preprocess_data
    raw_data = preprocess_data(np.array(dataset[0]), dataset[1], dataset[2], True)
    print("Preprocessing done")

    ## split
    train_test_split(raw_data[0], raw_data[1], 0.2)
    

if __name__ == "__main__":
    print("##################################################################################################")
    main(missings_check)

    






