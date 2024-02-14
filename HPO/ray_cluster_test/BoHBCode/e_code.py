# Add an introduction to the file as a multi-line comment
"""
Ekrems code to get the incumbent configuration
"""
import os
import json
import hpbandster.core.result as hpres
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def get_incumbent(_list):
    curr_incumbent = _list[0]
    incumbent_list = [curr_incumbent]
    for el in _list[1:]:
        if el < curr_incumbent:
            curr_incumbent = el
        incumbent_list.append(curr_incumbent)

    return incumbent_list


def get_incumbent_config(hpo_result_dir, config_savedir="experiments/", warmstart=False):
    datasets = [d for d in os.listdir(hpo_result_dir) if 'warmstart' not in d]

    completed_evaluations = dict()
    for dataset_name in datasets:

        all_prev_runs_max_budget = []

        if warmstart:
            res = hpres.logged_results_to_HBS_result(os.path.join(hpo_result_dir, dataset_name + '_warmstart'))
            res_prev = hpres.logged_results_to_HBS_result(os.path.join(hpo_result_dir, dataset_name))
            all_prev_runs_max_budget = [r for r in res_prev.get_all_runs() if r['loss'] != None]
            iter_n = len(all_prev_runs_max_budget) + 1
        else:
            res = hpres.logged_results_to_HBS_result(os.path.join(hpo_result_dir, dataset_name))
            iter_n = 1

        id2config = res.get_id2config_mapping()
        all_runs_max_budget = [r for r in res.get_all_runs() if r['loss'] != None]  # get the erronous runs out

        best_run = min(all_runs_max_budget, key=lambda x: x["loss"])
        best_config = id2config[best_run['config_id']]
        best_config = best_config['config']
        iter_n += best_run['config_id'][0]
        all_runs_max_budget = all_prev_runs_max_budget + all_runs_max_budget

        print(f'Dataset {dataset_name}')
        print(f'A total of {len(all_runs_max_budget)} runs were executed.')
        print(f'Best found configuration: {best_config}')
        print(f'Test accuracy: {1 - float(best_run.loss)}')
        print(f'On iteration {iter_n}')
        print(f'#######################################################################')

        best_config.update({"accuracy": 1 - float(best_run.loss)})

        if config_savedir is not None:
            config_run_history = os.path.join(config_savedir, "history")
            os.makedirs(config_savedir, exist_ok=True)
            os.makedirs(config_run_history, exist_ok=True)

            if 'warmstart' in dataset_name:
                dataset_name = dataset_name[:-10]

            run_incumbent = get_incumbent([r.loss for r in all_runs_max_budget])
            plt.plot(np.arange(len(run_incumbent)), run_incumbent)
            plt.savefig(os.path.join(config_run_history, dataset_name + '.png'))
            plt.clf()

            with open(os.path.join(config_savedir, dataset_name + '.json'), 'w') as f:
                json.dump(best_config, f)

        completed_evaluations[dataset_name] = len(all_runs_max_budget)

    return completed_evaluations


if __name__ == '__main__':
    hpo_result_dir = os.getcwd()
    config_savedir = os.path.join(os.getcwd(), "configs")

    get_incumbent_config(hpo_result_dir, config_savedir, True)