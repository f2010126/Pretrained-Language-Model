
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import xgboost as xgb
from tqdm import tqdm


# local imports
from metamodel_data import get_data_loader, preprocess_data

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

# seed everything
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

def compare_predictions(outputs, labels, threshold=0.3):
    count_within_threshold = 0
    for output, label in zip(outputs, labels):
        if abs(output - label) <= threshold:
            count_within_threshold += 1
    return count_within_threshold

## XGBoost
def create_dmatrix(loader):
    X, y = next(iter(loader))
    return xgb.DMatrix(X.numpy(), label=y.numpy())

def train_xgboost():
    training_set = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation/nlp_data_m.csv')
    X, y = preprocess_data(training_set)
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



class TrainModel():
    def __init__(self, input_size, hidden_size, output_size, epochs, lr, batch_size, fold_no, loss_func, seed):

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fold_no = fold_no
        self.loss_func = loss_func
        self.seed = seed
        # model
        self.model = MLP(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.train_loader=get_data_loader(batch_size, fold_no, 'train')
    

    def regression_training(self):
        self.model.train()
        batches=0
        running_loss = 0.0

        for x, acc, y_best in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = nn.MSELoss()(outputs, acc.unsqueeze(-1))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batches+=1

        # return an avg loss or something
        return running_loss/batches

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            
            if self.loss_func == "regression":
                avg_loss = self.regression_training()

            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
        
        return self.model

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data Creation")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--cv_fold', type=int, default=1, help='cv fold')
    parser.add_argument('--loss_func', type=str, default='regression', help='loss function can be regression|bpr')
    args = parser.parse_args()

    input_size = 27 # number of features encoded + dataset
    hidden_size = 64
    output_size = 1 # performance

    trainingObject=TrainModel(input_size, hidden_size, output_size, 10, 0.001, args.batch_size, args.cv_fold, args.loss_func, args.seed)
    model=trainingObject.train()
    
    # save the model
    torch.save(model.state_dict(), f'.metamodel_cvfold{args.cv_fold}_{args.loss_func}.pkl')

    # load_and_test(input_size, hidden_size, output_size)
    # train_xgboost()
