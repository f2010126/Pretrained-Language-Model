from math import e
from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from config2Vector import ENCODE_MODELS
import pandas as pd
import xgboost as xgb

optimizer_columns=['Adam','AdamW''SGD','RAdam']
scheduler_columns=['linear_with_warmup', 'cosine_with_warmup', 
                   'cosine_with_hard_restarts_with_warmup', 
                   'polynomial_decay_with_warmup', 'constant_with_warmup']
                
model_columns=["bert-base-uncased",
               "bert-base-multilingual-cased",
               "deepset/bert-base-german-cased-oldvocab",
               "uklfr/gottbert-base",
               "dvm1983/TinyBERT_General_4L_312D_de",
               "linhd-postdata/alberti-bert-base-multilingual-cased",
               "dbmdz/distilbert-base-german-europeana-cased"]

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def calculate_r_squared(outputs, labels):
    # Calculate R-squared
    mean_actual = torch.mean(labels)
    total_sum_of_squares = torch.sum((labels - mean_actual)**2)
    residual_sum_of_squares = torch.sum((outputs - labels)**2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared.item()

def preprocess_data():
    # convert meta data to include the one-hot encoding
    # returns the training, validation, and test sets
    training_set = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv')
    # remove the dataset and incumbent columns since its not metadata per se
    training_set = training_set.drop(columns=['Dataset', 'IncumbentOf'])
    model_dummies = pd.get_dummies(training_set['model'], prefix='Model')
    model_dummies = model_dummies.astype(int)
    optimizer_dummies = pd.get_dummies(training_set['optimizer'], prefix='Optimizer')
    optimizer_dummies = optimizer_dummies.astype(int)
    scheduler_dummies = pd.get_dummies(training_set['scheduler'], prefix='Scheduler')
    scheduler_dummies = scheduler_dummies.astype(int)

    training_set = pd.concat([training_set.drop(columns=['model','optimizer','scheduler']), 
                                      model_dummies,optimizer_dummies,scheduler_dummies], axis=1)
    # expands to 29 columns with one-hot encoding   
    # change the model, optimizer, and scheduler columns to one-hot encoding
    X = training_set.drop(columns=['Performance', 'Rank']) # 27 features 
    y = training_set['Performance']
    return X, y



    pass
def  load_process_data(batch_size=32, seed=42):
    X, y = preprocess_data()
    print(X.dtypes)
    # Preprocess the data
    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)  # 0.25 x 0.8 = 0.2

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader


def run_training(input_size, hidden_size, output_size, epochs=10, lr=0.001):
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader, test_loader = load_process_data(batch_size=32, seed=42)

    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_r_squared = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                val_loss += loss.item()
                val_r_squared += calculate_r_squared(outputs, labels)
            val_loss /= len(val_loader)
            val_r_squared /= len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation R-squared: {val_r_squared}")


    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_r_squared = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            test_r_squared += calculate_r_squared(outputs, labels)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        test_r_squared /= len(test_loader)
    print(f"Test Loss: {test_loss}, Test R-squared: {test_r_squared}")


def create_dmatrix(loader):
    X, y = next(iter(loader))
    return xgb.DMatrix(X.numpy(), label=y.numpy())

def train_xgboost():
    X, y = preprocess_data()
    X.columns=[col.replace('[', '').replace(']', '').replace('<', '') for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',             # Evaluation metric
    'eta': 0.1,                        # Learning rate
    'max_depth': 6,                    # Maximum depth of tree
    'min_child_weight': 1,             # Minimum sum of instance weight needed in a child
    'subsample': 0.7,                  # Subsample ratio of the training instance
    'colsample_bytree': 0.7,           # Subsample ratio of columns when constructing each tree
    'seed': 42                         # Random seed
    }
    num_rounds = 100  # Number of boosting rounds
    eval_results = {}
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, num_rounds, evals=evals, early_stopping_rounds=10)
    # Evaluate the model on train, validation, and test sets
    preds = model.predict(dtest)
    y_label = dtest.get_label()
    mse = mean_squared_error(y_label, preds)
    print(f"Mean Squared Error on Test set: {mse}")


if __name__ == "__main__":
    input_size = 27 # number of features encoded + dataset
    hidden_size = 64
    output_size = 1 # performance
    # run_training(input_size, hidden_size, output_size,epochs=100, lr=0.001)
    train_xgboost()
