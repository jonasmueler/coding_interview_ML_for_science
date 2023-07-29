import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import *
from XGBoost import get_data

## 
import os

class MLP(nn.Module):
    def __init__(self, input_size: 20, output_size: 1, hidden_layer_size: int, hidden_layers: int):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = hidden_layers

        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_size, self.hidden_layer_size))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_layer_size))

        # Hidden layers
        for i in range(self.hidden_layers):
            layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.hidden_layer_size))
            

        # Output layer
        layers.append(nn.Linear(self.hidden_layer_size, self.output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

"""    
## test 
model = MLP(20, 1, 3, 5)
test = torch.rand(20,20) # (batch, input)
assert model(test).shape == (20,1)
""" 
   
class Dataset(Dataset):
    """
    Dataset class necessary for DataLoader class
    """
    def __init__(self, X_data: np.ndarray, y_data: np.array):
        self.X_data = torch.from_numpy(X_data)
        self.y_data = torch.from_numpy(y_data) 

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        
        return self.X_data[idx], self.y_data[idx]
    
def get_data_loader(dataset: Dataset, batch_size: int):
    """
    Creates DataLoader object for stochstic neural network optimization 
    """
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    return data_loader

def trainLoop(train_loader: DataLoader, val_loader: DataLoader, model: MLP, criterion: torch.nn, criterion_val: torch.nn, params: dict, safe_descent: bool):
    """
    Train Loop to optimize model parameters
    """
    
    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = params["learning_rate"],
                                      weight_decay= params["weight_decay"])
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr = params["learning_rate"],
                                      weight_decay= params["weight_decay"])

    if safe_descent: # optimize if architecture becomes larger, pre-init memory 
        val_losses = []

    # Early stopping parameters
    patience = params["patience"]  
    best_val_loss = float('inf')
    epochs_without_improvement = 0


    counter = 0
    for b in range(params["epochs"]):
        for inpts, targets in train_loader:
            model.train()

            inpts.float()
            targets.float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            forward = model(inpts)
            #print(forward)
            #print(forward.size())
            #print(targets.size())
            loss = criterion(forward.squeeze(), targets.squeeze()) 
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0) # gradient clipping; no exploding gradient
            optimizer.step()
            counter += 1
            #print(loss)
               
        ## validation performance after each epoch
        for inpts, targets in val_loader:
            model.eval()
            X, y = next(iter(val_loader))
            
            # predict
            pred = model.forward(X)
            val_loss = criterion_val(pred, y)

        print(f"epoch {b} Val Score: {val_loss}")

        if safe_descent:
            os.chdir(path_origin)
            path = os.path.join(path, "results")
            os.makedirs(path, exist_ok = True)
            result = np.array(val_losses)
            np.savetxt("val_results_best_model.csv", result, delimiter=",")
            os.chdir(path)

        """
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Validation loss did not improve for {patience} epochs. Early stopping.")
                # save parameters a and score
                return [params["learning_rate"], params["weight_decay"], params["batch_size"], best_val_loss]
        """
        

def acc (pred: torch.tensor, truth: torch.tensor):

    # get predictions of model 
    acc = (pred > 0.5).float()
    
    acc = torch.sum(acc == truth)/len(pred)
    

    return acc
                                  
if __name__ == "__main__":
    # get the data 
    X_train, y_train, X_test, y_test = get_data()

    # split train for validtaion set and train set
    val_criterion = 0.90 
    train_dataset = Dataset(X_train[0:round(val_criterion*len(X_train)),:], y_train[0:round(val_criterion*len(y_train)),:])
    val_dataset = Dataset(X_train[round(val_criterion*len(X_train)):, :], y_train[round(val_criterion*len(X_train)):,:])

    # define hyperparameters relu lr = 0.0001
    hyper = {"optimizer": "adam", "learning_rate" : 0.0001, "weight_decay": 0.01, "patience": 5, "batch_size": 100, "epochs": 10000}

    # get dataloaders
    data_loader_train = DataLoader(train_dataset, batch_size=hyper["batch_size"], shuffle = True)
    data_loader_val = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = True)
    
    # define model 
    net = MLP(input_size = 20, output_size = 1, hidden_layer_size = 100, hidden_layers =2).float()
    
    # criteria
    cross_entropy = torch.nn.BCELoss()
    F1_score = acc

    # train network
    trainLoop(data_loader_train, data_loader_val, net, cross_entropy, F1_score, hyper, False) 



