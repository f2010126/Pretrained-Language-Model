import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

import argparse
import logging
import multiprocessing
import os

def realtime_learning_curves(runs):
    """
    example how to extract a different kind of learning curve.

    The x values are now the time the runs finished, not the budget anymore.
    We no longer plot the validation loss on the y axis, but now the test accuracy.

    This is just to show how to get different information into the interactive plot.

    """
    sr = sorted(runs, key=lambda r: r.budget)
    lc = list(filter(lambda t: not t[1] is None, [(r.time_stamps['finished'], r.info['test accuracy']) for r in sr]))
    return([lc,])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View Curves')
    parser.add_argument('--dir', type=str, help='Directory of the results.', default='datasetruns/bohb_gnad10_seed_9_150_trials')
    args = parser.parse_args()

    # where all the run artifacts are kept
    working_dir = os.path.join(os.getcwd(), args.dir)


    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(working_dir)
    # get all executed runs

    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    # inc_test_loss = inc_run.info['test accuracy']

    print('Best found configuration:')
    print(inc_config)
   # print('It achieved accuracies of %f (validation) and %f (test).' % (1 - inc_loss, inc_test_loss))

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    plt.show()
