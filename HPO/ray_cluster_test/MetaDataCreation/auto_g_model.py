import time
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import sys
sys.path.append(os.path.abspath('/work/dlclarge1/dsengupt-zap_hpo_og/TinyBert/HPO/ray_cluster_test'))
from BoHBCode.data_modules import get_datamodule
from BoHBCode.evaluate_single_config import train_single_config
from generate_perfmatrix import evaluate_config

def train():
    train_data = TabularDataset(f'nlp_data_m.csv')
    train_data.head()
    label = 'Performance'
    fit_param=60*60
    predictor = TabularPredictor(label=label,verbosity=3,
                                 eval_metric='root_mean_squared_error').fit(
                                     train_data.drop(columns=["Rank"]),
                                     ag_args_fit={'num_gpus': 1}, 
                                     time_limit=fit_param)
    test_data = TabularDataset(f'test_germeval2018.csv')
    y_pred = predictor.predict(test_data)
    print(y_pred)
    print('End')

def test_model(seed=42):
    label = 'Performance'
    predictor = TabularPredictor.load("AutogluonModels/ag-20240423_172315")
    test_data = TabularDataset(f'test_germeval2018.csv')
    y_pred = predictor.predict(test_data.drop(columns=["Rank"]))
    max_index = y_pred.idxmax()
    # model name
    model_name=test_data.loc[max_index]['IncumbentOf']
    print(f"Incumbent model: {model_name}")
    predictor.evaluate(test_data, silent=True)
    print('End')

if __name__ == "__main__":
    train()
