import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import *
from XGBoost import get_data
import sklearn

## 
import os # save things in folders
import time # check runtime

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
        layers.append(nn.LayerNorm(self.hidden_layer_size))
        #layers.append(nn.Dropout(p = 0.5))
        layers.append(nn.GELU())

        # Hidden layers
        for i in range(self.hidden_layers):
            layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            #layers.append(nn.BatchNorm1d(self.hidden_layer_size))
            layers.append(nn.LayerNorm(self.hidden_layer_size))
            #layers.append(nn.Dropout(p = 0.5))
            layers.append(nn.GELU())
            

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
    def __init__(self, X_data: np.ndarray, y_data: np.array) -> None:
        self.X_data = torch.from_numpy(X_data)
        self.y_data = torch.from_numpy(y_data) 

    def __len__(self)-> int:
        return len(self.X_data)

    def __getitem__(self, idx: int) -> np.ndarray:
        
        return self.X_data[idx], self.y_data[idx]
    
def get_data_loader(dataset: Dataset, batch_size: int)-> DataLoader:
    """
    Creates DataLoader object for stochstic neural network optimization 
    """
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    return data_loader

def trainLoop(train_loader: DataLoader, 
              val_loader: DataLoader, 
              model: MLP, 
              criterion: torch.nn, 
              criterion_val: torch.nn, 
              params: dict, 
              safe_descent: bool,
              safe_model: bool = False) -> list():
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
    best_val_loss = 0
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

        #print(f"epoch {b} Val Score: {val_loss}")

        if safe_descent:
            os.chdir(path_origin)
            path = os.path.join(path, "results")
            os.makedirs(path, exist_ok = True)
            result = np.array(val_losses)
            np.savetxt("val_results_best_model.csv", result, delimiter=",")
            os.chdir(path)

        
        # Check for early stopping
        if val_loss >= best_val_loss:
            best_val_loss = val_loss.detach().cpu()
            epochs_without_improvement = 0

            # reset early stopping counter
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                #print(f"Validation loss did not improve for {patience} epochs. Early stopping.")
                # save parameters and score

                # save final optimized model
                if safe_model:
                    # create folder
                    path = os.path.join(path_origin, "models", "optimizd_MLP.pth")
                    model.eval()
                    torch.save(model.state_dict(), path)
                    print("Model saved!")
                    os.chdir(path_origin)


                return [params["learning_rate"], 
                        params["weight_decay"], 
                        params["batch_size"], 
                        params["hidden_layer_size"], 
                        params["hidden_layers"], 
                        val_loss.item()]
        
        

def acc (pred: torch.tensor, truth: torch.tensor) -> torch.tensor: 
    """
    calculate accuracy from predicted class probabilities and ground truth labels
    """

    # get predictions of model 
    acc = (pred > 0.5).float()
    
    # calculate acc
    acc = torch.sum(acc == truth)/len(pred)

    return acc

def f1_score(pred: torch.tensor, truth: torch.tensor) -> torch.tensor: 
    """
    calculate f1 score from predicted class probabilities and ground truth labels
    """
    # get function from sklearn
    f1 = sklearn.metrics.f1_score

    # get predictions of model 
    class_pred = (pred > 0.5).float()
    class_pred = class_pred.numpy()
    truth = truth.numpy()

    # predict
    score = f1(class_pred, truth)

    # back to torcvh fro train loop
    score = torch.tensor(score)

    return score

def random_search_MLP(n_iter: int, final_model: bool) -> np.ndarray:
    """
    Implements random search over the selected hyperparameters:
    learning rate
    weight decay
    batch_size
    hidden_layer_size 
    hidden_layers
    """
    print("Start Random Search Optimization for Neural Network")
    # get the data 
    X_train, y_train, _, _ = get_data()

    # split train for validtaion set and train set
    val_criterion = 0.90 
    train_dataset = Dataset(X_train[0:round(val_criterion*len(X_train)),:], y_train[0:round(val_criterion*len(y_train)),:])
    val_dataset = Dataset(X_train[round(val_criterion*len(X_train)):, :], y_train[round(val_criterion*len(X_train)):,:])
    train_dataset_full = Dataset(X_train, y_train)

    # init memory
    results = np.zeros((n_iter, 6)) 
    best_combination = np.zeros((n_iter, 6)) 
    print("start searching!")

    # start searching
    for i in range(n_iter):
        # define hyperparameters 0.0001
        hyper_parameters = {"optimizer": "adam", 
                            "learning_rate" : np.random.uniform(0.00001, 0.001), 
                            "weight_decay": np.random.uniform(0.00001, 0.01), 
                            "patience": patience, 
                            "batch_size": np.random.randint(10, 50), 
                            "epochs": 1000,
                            "hidden_layer_size": np.random.randint(5, 30), 
                            "hidden_layers": np.random.randint(2, 8)}
        
        # get dataloaders
        data_loader_train = DataLoader(train_dataset, batch_size = hyper_parameters["batch_size"], shuffle = True)
        data_loader_val = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = True)

        # extra data loader to train final model on all data
        if final_model == True:
            data_loader_full = DataLoader(train_dataset_full, batch_size = hyper_parameters["batch_size"], shuffle = True)

        # define model 
        net = MLP(20, 1, hyper_parameters["hidden_layer_size"], hyper_parameters["hidden_layers"]).float()
    
        # criteria
        cross_entropy = torch.nn.BCELoss()
        accuracy = f1_score

        res = trainLoop(data_loader_train, data_loader_val, net, cross_entropy, accuracy, hyper_parameters, False)

        # safe results 
        results[i,:] = res
        best = np.argmax(results[:,-1])
        best_par = results[best, :]
        best_combination[i, :] = best_par
        
        print(f"Iteration {i + 1} done")
        print(f"Current best parameter-set: lr = {best_par[0]}, \
              weight-decay = {best_par[1]}, \
              batch_size = {best_par[2]}, \
              layer-size = {best_par[3]}, \
              layers = {best_par[4]}, \
              F1-validation-score = {best_par[5]}")
    
    # save optimization data 
    # add time column 
    results = np.c_[results, np.arange(0,len(results[:,0]))]
    best_combination = np.c_[best_combination, np.arange(0,len(best_combination[:,0]))]

    # save 
    os.chdir(path_origin)
    path = os.path.join(path_origin, "optimization_results_MLP")
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    np.save("optimization_combinations.npy", results)
    np.save("best_parameter_combination_over_time.npy", best_combination)
    print("Data saved!")
    os.chdir(path_origin)


    # train and save final model 
    if final_model:
        hyper_parameters = {"optimizer": "adam", 
                            "learning_rate" : best_par[0], 
                            "weight_decay": best_par[1], 
                            "patience": patience, 
                            "batch_size": best_par[2], 
                            "epochs": 10000,
                            "hidden_layer_size": best_par[3], 
                            "hidden_layers": best_par[4]}
        #model 
        net = MLP(20, 1, int(hyper_parameters["hidden_layer_size"]), int(hyper_parameters["hidden_layers"])).float()
        trainLoop(data_loader_full, data_loader_val, net, cross_entropy, accuracy, hyper_parameters, False, safe_model = final_model)
        
    return results

                                 
if __name__ == "__main__":
    print("##################################################################################################")
    start = time.time()
    res = random_search_MLP(n_iter, True)
    end = time.time()
    print("Elapsed time:",)
    print(f"{(end-start)/60} minutes")
    


