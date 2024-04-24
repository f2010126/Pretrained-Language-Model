"""
Plots based on stuff from the MetaDataCreation folder

"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

def save_csv(df, name):
    """
    Save a dataframe to a csv file

    :param df: Dataframe
    :param name: Name of the file
    """
    try:
        df.to_csv(name + ".csv", index=False)
    except Exception as e:
        print("Error saving csv file: ", e)

def sort_best(df):
    """
    Sort the dataframe by best across all datasets
    """
    # add up across all datasets then average, best one is the one with the highest average
    sum_performance = df.sum(axis=0)
    average_performance = sum_performance / len(df.index)
    df = df[average_performance.sort_values(ascending=False).index]
    return df



def generate_heatmap(perf_matrix, title):
    """
    Generate a heatmaps

    :param perf_matrix:performance matrix
    :param title: Title
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(perf_matrix, cmap='magma', cbar=True, fmt=".6f")
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate the performance matrix from the run results')
    parser.add_argument('--og_perf', type=str, help='Location of the original performance matrix',default='heatmap_OG_performance_matrix.csv')
    parser.add_argument('--new_perf', type=str, help='Name of the new performance matrix',default='best_sorted_performance_matrix.csv')
    args = parser.parse_args()

    og_perf = pd.read_csv('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/heatmap_OG_performance_matrix.csv', index_col=0)
    sorted_perf = sort_best(og_perf)
    generate_heatmap(sorted_perf, "Best")
    

