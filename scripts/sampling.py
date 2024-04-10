import pandas as pd
import numpy as np

def simple_random_sampling(df, n=None, fraction=None, seed=None):
    """
    Perform simple random sampling on a DataFrame.

    :param df: Pandas DataFrame to sample from.
    :param n: Number of samples to draw.
    :param fraction: Fraction of samples to draw.
    :param seed: Random seed for reproducibility.
    :return: Sampled DataFrame.
    """
    if n is not None:
        return df.sample(n, random_state=seed)
    elif fraction is not None:
        return df.sample(frac=fraction, random_state=seed)
    else:
        raise ValueError("Specify either n (number of samples) or fraction (fraction of samples)")

def systematic_sampling(df, n, seed=None):
    """
    Perform systematic sampling on a DataFrame.

    :param df: Pandas DataFrame to sample from.
    :param n: Number of samples to draw.
    :param seed: Random seed for starting index.
    :return: Sampled DataFrame.
    """
    if n is None or n <= 0:
        raise ValueError("Specify a positive value for n (number of samples)")
    
    interval = len(df) // n
    start = np.random.randint(0, interval) if seed is None else seed % interval
    indices = np.arange(start, len(df), interval)
    return df.iloc[indices]

def stratified_sampling(df, strata, n=None, fraction=None, seed=None, stratified_type='proportionate', strata_n=None):
    """
    Perform stratified sampling on a DataFrame (Single-Stage).

    :param df: Pandas DataFrame to sample from.
    :param strata: Column name for stratification.
    :param n: Total number of samples to draw (for proportionate and disproportionate sampling).
    :param fraction: Fraction of samples to draw (for proportionate stratified sampling).
    :param seed: Random seed for reproducibility.
    :param stratified_type: Type of stratified sampling ('proportionate' or 'disproportionate').
    :param strata_n: Dictionary specifying the number of samples from each stratum (for disproportionate sampling).
    :return: Sampled DataFrame.
    """
    if strata not in df.columns:
        raise ValueError(f"{strata} column not found in DataFrame")

    if stratified_type == 'proportionate':
        if fraction is None:
            raise ValueError("For proportionate stratified sampling, specify fraction")
        return df.groupby(strata, group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=seed))

    elif stratified_type == 'disproportionate':
        if strata_n is None or not isinstance(strata_n, dict):
            raise ValueError("For disproportionate stratified sampling, provide strata_n as a dictionary")
        return df.groupby(strata, group_keys=False).apply(lambda x: x.sample(n=strata_n.get(x.name, 0), random_state=seed))

    else:
        raise ValueError("Invalid stratified_type. Choose from 'proportionate', 'disproportionate'")
