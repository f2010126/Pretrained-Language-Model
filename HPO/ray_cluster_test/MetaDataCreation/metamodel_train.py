
import argparse
import re
import sched
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import ndcg_score

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
        self.sigmoid = nn.Sigmoid() # get values between 0 and 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  #squash
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
    def __init__(self, input_size, hidden_size, output_size, epochs, lr, batch_size, fold_no, loss_func, seed,config=None):

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fold_no = fold_no
        self.loss_func = loss_func
        self.seed = seed
        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_size, hidden_size, output_size).to(self.device)

        if config['optimizer_type'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif config['optimizer_type'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=config['sgd_momentum'])
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        if config['scheduler_type'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif config['scheduler_type'] == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        else:
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)
    
        self.criterion = nn.MSELoss()
        self.config=config

        self.train_loader=get_data_loader(batch_size=batch_size, cv_fold=fold_no, seed=self.seed,loss_func=self.loss_func)
    

    def regression_training(self):
        self.model.train()
        batches=0
        running_loss = 0.0

        for x, acc, y_best in self.train_loader:
            x, acc = x.to(self.device), acc.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(x)
            loss = nn.MSELoss()(outputs, acc.unsqueeze(-1))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batches+=1

        # return an avg loss or something
        return running_loss/batches

    def hingeloss_training(self):
        self.model.train()
        batches = 0
        running_loss = 0.0

        for (x, s, l), accuracies, ranks in self.train_loader:
            x, s, l = x.to(self.device), s.to(self.device), l.to(self.device)
            self.optimizer.zero_grad()
        
            outputs = self.model.forward(x)
            outputs_small = self.model.forward(s)
            outputs_larger = self.model.forward(l)

       
            hinge_loss = nn.MarginRankingLoss(margin=1.0)
            targets = torch.ones(outputs.shape[0], dtype=torch.float32).unsqueeze(-1)  # Positive pairs
            loss = hinge_loss(outputs, outputs_small, targets)  # Loss for smaller performance configurations
            loss += hinge_loss(outputs, outputs_larger, -targets)  # Loss for larger performance configurations

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batches += 1

        return running_loss / batches

    def bpr_training(self):
        self.model.train()
        batches=0
        running_loss = 0.0

        for (x, s, l), accuracies, ranks in self.train_loader:
            x, s, l = x.to(self.device), s.to(self.device), l.to(self.device)
            self.optimizer.zero_grad()
            
            # Perf predictions for target, inferior, superior configurations.
            outputs = self.model.forward(x)
            outputs_small = self.model.forward(s)
            outputs_larger = self.model.forward(l)

            output_gr_smaller = nn.Sigmoid()(outputs - outputs_small)
            larger_gr_output  = nn.Sigmoid()(outputs_larger - outputs)
            larger_gr_smaller  = nn.Sigmoid()(outputs_larger - outputs_small)

            logits = torch.cat([output_gr_smaller,larger_gr_output,larger_gr_smaller], 0)
            loss = nn.BCELoss()(logits, torch.ones_like(logits).to(self.device))

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batches+=1

        # return an avg loss or something
        return running_loss/batches

    def validation(self,use_set="valid"):
        self.model.eval()
        # only for evaluation purposes, change the loss function to regression to load the data  in the correct format
        self.train_loader.dataset.loss_func='regression'
        og_set = self.train_loader.dataset.set
        self.train_loader.dataset.set=use_set
        y_true = []
        y_pred = []
        y_score = []
        with torch.no_grad():
            for x, acc, y_best in self.train_loader:
                x, acc = x.to(self.device), acc.to(self.device)
                outputs = self.model(x)

                # calculate probabilities from outputs
                probabilities = torch.sigmoid(outputs)
                # line the outputs and labels up
                y_true.extend(acc.cpu().numpy())

                y_score.extend(probabilities.cpu().numpy())
                if isinstance(outputs.squeeze().tolist(), float):
                    y_pred.append(outputs.squeeze().tolist())
                else:
                    y_pred.extend(outputs.squeeze().tolist())
            
            # Convert lists to numpy arrays
        y_true = np.array(y_true)
        y_score = np.squeeze(y_score)
        ndcg1=ndcg_score([y_true], [y_score],k=1)
        ndcg5 = ndcg_score([y_true], [y_score],k=5)
        ndcg10 = ndcg_score([y_true], [y_score],k=10)
        ndcg20 = ndcg_score([y_true], [y_score],k=20)
        # print(f" Set {use_set} Validation NDCG@5: {ndcg5}, NDCG@10: {ndcg10}, NDCG@20: {ndcg20}")

        # after validation, set the loss function back to whatever it was in the train object
        self.train_loader.dataset.loss_func=self.loss_func
        self.train_loader.dataset.set=og_set
        return ndcg1
        
    def test(self):
        return self.validation(use_set="test")

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            
            # Training
            self.train_loader.dataset.set="train"
            if self.loss_func == "regression":
                avg_loss = self.regression_training()
            elif self.loss_func == "bpr":
                avg_loss = self.bpr_training()
            elif self.loss_func == "hingeloss":
                avg_loss = self.hingeloss_training()

            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
            self.scheduler.step()

            # Validation
            ndcg1_train= self.validation(use_set="train")
            ndcg1_valid= self.validation(use_set="valid")
            
        return self.model, ndcg1_valid

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data Creation")
    parser.add_argument('--batch_size', type=int, default=204, help='batch size should be number of pipelines in the dataset')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--cv_fold', type=int, default=3, help='cv fold')
    parser.add_argument('--loss_func', type=str, default='hingeloss', help='loss function can be regression|bpr|hingeloss')
    args = parser.parse_args()

    input_size = 27 # number of features encoded + dataset
    hidden_size = 64
    output_size = 1 # performance

    trainingObject=TrainModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                              epochs=3, lr=0.0001, batch_size=args.batch_size, fold_no=args.cv_fold, 
                              loss_func=args.loss_func, seed=args.seed)
    model, ndcg1_val=trainingObject.train()
    test_ndcg1=trainingObject.test()
    print(f"Validation NDCG@1: {ndcg1_val}, Test NDCG@1: {test_ndcg1}")
    
    # save the model
    torch.save(model.state_dict(), f'.metamodel_cvfold{args.cv_fold}_{args.loss_func}.pkl')

    # load_and_test(input_size, hidden_size, output_size)
    # train_xgboost()
