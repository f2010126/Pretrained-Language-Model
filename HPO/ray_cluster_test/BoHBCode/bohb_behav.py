"""
Observe behavior of BOHB across different hyperparameters. taken from an issue on the hpbandster github.
"""
import threading
import numpy as np
import Pyro4
import Pyro4.naming

from hpbandster.core.result import Result
from hpbandster.core.base_iteration import Datum

# Add documenation for the function
"""
Observes the behavior of BOHB across different hyperparameters.
:param min_budget: the smallest budget to consider
:param max_budget: the largest budget to consider
:param eta: the eta parameter. Determines how many configurations advance to the next round
:param n_iterations: how many iterations of SuccessiveHalving to perform
"""


def predict_bohb_run(min_budget, max_budget, eta, n_iterations):
    """
        Prints the expected numbers of configurations, runs and budgets given BOHB's hyperparameters.

        Parameters
        ----------
        min_budget : float
            The smallest budget to consider.
        max_budget : float
            The largest budget to consider.
        eta : int
            The eta parameter. Determines how many configurations advance to the next round.
        n_iterations : int
            How many iterations of SuccessiveHalving to perform.
        """
    s_max = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1

    n_runs = 0
    n_configurations = []
    initial_budgets = []
    for iteration in range(n_iterations):
        s = s_max - 1 - (iteration % s_max)

        initial_budget = (eta ** -s) * max_budget
        initial_budgets.append(initial_budget)

        n0 = int(np.floor(s_max / (s + 1)) * eta ** s)
        n_configurations.append(n0)
        ns = [max(int(n0 * (eta ** (-i))), 1) for i in range(s + 1)]
        n_runs += sum(ns)

    print('Running BOHB with these parameters will proceed as follows:')
    print('  {} iterations of SuccessiveHalving will be executed.'.format(n_iterations))
    print('  The iterations will start with a number of configurations as {}.'.format(n_configurations))
    print('  With the initial budgets as {}.'.format(initial_budgets))
    print('  A total of {} unique configurations will be sampled.'.format(sum(n_configurations)))
    print('  A total of {} runs will be executed.'.format(n_runs))
    # print newline
    print()


def try_predict_bobh_run(min_budget=1, max_budget=10, eta=3, n_iterations=70):
    result = predict_bohb_run(min_budget=min_budget, max_budget=max_budget, eta=eta, n_iterations=n_iterations)
    print(result)

if __name__ == "__main__":
    try_predict_bobh_run(min_budget=1, max_budget=2, eta=2, n_iterations=2)

    # for i in range(1, 14):
    #     print(f'Iteration {i}')
    #     try_predict_bobh_run(min_budget=1, max_budget=12, eta=12, n_iterations=i)

    print('Done')