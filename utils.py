from multiprocessing import Pool

import os
import pandas as pd
import numpy as np
import pickle as pk
from tqdm import tqdm
from os.path import join as path_join
import time
import random
import sys
import hashlib
import scipy.stats as stats


def pickelize(obj,file):
    with open(file,"wb") as f:
        pk.dump(obj,f)

def un_pickelize(file):
    with open(file,"rb") as f:
        return(pk.load(f)) 




from scipy import stats

def mannwhitneyu(S1, S2, alpha=0.1):
    """
    Compare two independent samples to determine if the difference between them is significant.

    Parameters:
    S1 (array-like): The first sample.
    S2 (array-like): The second sample.
    alpha (float): Significance level, default is 0.05.

    Returns:
    str: 'not significant', 'larger', or 'smaller', depending on the test results.
    """
    # if the size of each is zero return not significant
    if len(S1) == 0 or len(S2) == 0:
        return 0
    # Perform the Mann-Whitney U Test
    stat, p_value = stats.mannwhitneyu(S1, S2, alternative='two-sided')

    # Check if the result is significant
    if p_value > alpha:
        return 0

    # Determine which sample is larger
    if stat < len(S1) * len(S2) / 2:
        return 1
    else:
        return -1



import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cdf_ccdf(
    values, 
    xlabel="X", 
    ylabel="CDF", 
    file_name="plot", 
    save_dir="./figs/cdf_ccdfs", 
    font="Times New Roman", 
    font_size=24,
    gridlines=None,
    show_plot=True
):
    """
    Plots the CDF and CCDF  on a dual y-axis log-log plot.

    Parameters:
        values (array-like): Data to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the CDF y-axis.
        file_name (str): Name of the output file (without extension).
        save_dir (str): Directory to save the output plot.
        font (str): Font to use in the plot.
        font_size (int): Font size.
        gridlines (dict): Optional dictionary with keys like 'x_cdf', 'y_cdf', 'x_ccdf', 'y_ccdf'.
        show_plot (bool): Whether to display the plot with plt.show().
    """

    # Set global font config
    plt.rcParams.update({
        'font.family': font,
        'pdf.fonttype': 42,
        'font.size': font_size
    })

    def empirical_cdf(data):
        sorted_data = np.sort(data)
        n = len(data)
        return sorted_data, np.arange(1, n + 1) / n

    def empirical_ccdf(data):
        sorted_data, cdf_data = empirical_cdf(data)
        return sorted_data, 1 - cdf_data + 1 / len(data)

    sorted_data, y_cdf = empirical_cdf(values)
    _, y_ccdf = empirical_ccdf(values)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # CDF
    ax1.plot(sorted_data, y_cdf, 'b-', linewidth=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('CDF', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # CCDF on secondary axis
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.plot(sorted_data, y_ccdf, 'r-', linewidth=2)
    ax2.set_ylabel('CCDF', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Log scale on x-axis
    ax1.set_xscale("log")

    # Optional grid lines
    if gridlines:
        for x in gridlines.get('x_cdf', []):
            ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.8, linewidth=0.6)
        for y in gridlines.get('y_cdf', []):
            ax1.axhline(y=y, color='gray', linestyle='--', alpha=0.8, linewidth=0.6)
        for x in gridlines.get('x_ccdf', []):
            ax2.axvline(x=x, color='gray', linestyle='--', alpha=0.8, linewidth=0.6)
        for y in gridlines.get('y_ccdf', []):
            ax2.axhline(y=y, color='gray', linestyle='--', alpha=0.8, linewidth=0.6)

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"cdf_ccdf_{file_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()



def y_fmt(tick_val, decimals=1):
    """
    Format tick values with human-readable suffixes (K, M, B).
    
    Parameters:
        tick_val (float): The tick value to format.
        decimals (int): Number of decimal places for formatting small values or scaled numbers.

    Returns:
        str: Formatted string with appropriate suffix.
    """
    abs_val = abs(tick_val)
    
    if abs_val >= 1e9:
        val = tick_val / 1e9
        suffix = 'B'
    elif abs_val >= 1e6:
        val = tick_val / 1e6
        suffix = 'M'
    elif abs_val >= 1e3:
        val = tick_val / 1e3
        suffix = 'K'
    else:
        return f"{tick_val:.{decimals}f}"

    return f"{val:.{decimals}f}{suffix}"


