import warnings

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac.facade import MultiFidelityFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


if __name__ == "__main__":
    from ConfigSpace import Configuration, ConfigurationSpace

    import numpy as np
    from smac import HyperparameterOptimizationFacade, Scenario
    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    iris = datasets.load_iris()


    def train(config: Configuration, seed: int = 0) -> float:
        classifier = SVC(C=config["C"], random_state=seed)
        scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
        return 1 - np.mean(scores)


    configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=200)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()