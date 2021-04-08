import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import pickle
import joblib
from collections import Counter
from datetime import datetime
from fancyimpute import SoftImpute, BiScaler
import softimpute_als

def imputation_cycles(df_dat, n = 100, m = 0.2, J = 20, save_dir = './imputations', seed = 543):
    """
    Generate and save n-cycles of imputation over m-fraction of data.
    :param df_dat: Assessment data
    :param n: Number of imputation cycles.
    :param m: Fraction of data to set to nan per cycle.
    :param J: SoftImpute-ALS J, rank to use to impute.
    :param save_dir: Folder to save data to (must exist).
    :param seed: Random seed.
    """
    np.random.seed(seed)
    n, m = 100, 0.2
    E, N = None, None
    for i in range(n):
        # Worry line
        print((i, datetime.now()))
        # Copy data array
        X = df_dat.values.copy()
        # Randomly set nan values in each column
        for c in range(X.shape[1]):
            X[np.random.randint(0, len(df_dat), int(m * len(df_dat))), c] = float('nan')
        # Impute nan
        Y = softimpute_als.SoftImpute(J = J).fit(X).predict(X)
        # Pickle result
        with open(f"./{save_dir}/imp_{i}.pkl", 'wb') as p:
            joblib.dump({'X': X, 'Y': Y}, p, compress='zlib')
        # Initialize E, N
        if E is None:
            E = np.zeros(X.shape)
            N = np.zeros(X.shape)
        # Keep sign of error
        E[np.isnan(X)] += (df_dat.values[np.isnan(X)] - Y[np.isnan(X)])
        N[np.isnan(X)] += 1

    with open(f"./{save_dir}/errors.pkl", 'wb') as p:
        joblib.dump({'E': E, 'N': N}, p, compress='zlib')

def conditional_errors(df_dat, tar = 'LOCALTOTAL_N', save_dir = './imputations'):
    """
    Calculate and save conditional errors, Y: MSE(tar, col is not na), N: MSE(tar, col is na)
    :param df_dat: Assessment data.
    :param tar: Target column
    :param save_dir: Folder to read from and save data to (must exist).
    """
    # Load conditional errors
    ic = df_dat.columns.to_list().index(tar)
    sqe_Y, sqe_N = None, None
    num_Y, num_N = None, None
    for i in range(n):
        print((i, datetime.now()))
        # Open this sample
        with open(f"./{save_dir}/imp_{i}.pkl", 'rb') as p:
            d = joblib.load(p)
        X = d['X']
        Y = d['Y']
        err = (df_dat.values[:, ic] - Y[:, ic]) ** 2
        # Initialize sqe/num first time
        if sqe_Y is None:
            sqe_Y = np.zeros(X.shape[1])
            sqe_N = np.zeros(X.shape[1])
            num_Y = np.zeros(X.shape[1])
            num_N = np.zeros(X.shape[1])
        # Loop over columns and calculate conditional error    
        for j in range(X.shape[1]):
            if j != ic:
                I_Y = (np.isnan(X[:, ic]) & ~np.isnan(X[:,j]))
                I_N = (np.isnan(X[:, ic]) &  np.isnan(X[:,j]))
                sqe_Y[j] += err[I_Y].sum() 
                sqe_N[j] += err[I_N].sum()
                num_Y[j] += I_Y.sum() 
                num_N[j] += I_N.sum()
        
    with open(f"./{save_dir}/errors.pkl", 'wb') as p:
        joblib.dump({'sqe_Y': sqe_Y, 'sqe_N': sqe_N, 'num_Y': num_Y, 'num_N': num_N}, p, compress='zlib')
    
    
def assessment_etl(assessment_csv = "./data/assessments.csv",
                   data_dict_csv = "./data/property-assessments-data-dictionary.csv",
                  ):
    """
    Construct working dataframe of assessment data from raw data.
    :param assessment_csv: Assessment data csv.
    :param data_dict_csv: Data dictionary csv (not the original, augmented with type notation).
    :return df_dat: Encoded and scaled assessment data.
    """
    
    # Data dictionary has spaces in column names, remove those
    df_pad = pd.read_csv(data_dict_csv, encoding = "ISO-8859-1")
    df_pad.rename(columns = {c: c.replace(' ', '') for c in df_pad.columns}, inplace = True)

    # Use data dictionary types to convert numbers make everything else a string
    dt = {c.FieldName: (float if c.DataTypeandLength.lower().startswith('n') else str) for c in df_pad.itertuples()}
    df_asm = pd.read_csv(assessment_csv, dtype=dt)
    
    # Extract single family homes
    df_asm = df_asm[(df_asm.CLASSDESC == 'RESIDENTIAL') & (df_asm.USEDESC == 'SINGLE FAMILY')].copy()
    df_asm.reset_index(drop = True, inplace = True)
    
    # Delete the county assessment (keeping LOCAL)
    del df_asm['COUNTYBUILDING']
    del df_asm['COUNTYLAND']
    del df_asm['COUNTYTOTAL']
    
    df_ohc = one_hot_columns(df_asm[[c for c in df_pad[df_pad.handle == 'oh'].FieldName]])
    df_dat = date_columns(df_asm[[c for c in df_pad[df_pad.handle == 'date'].FieldName]])
    df_num = numerical_columns(df_asm[[c for c in df_pad[df_pad.handle == 'num'].FieldName if c in df_asm]])
    df_bin = pd.DataFrame({'HOMESTEADFLAG_B': np.where(df_asm.HOMESTEADFLAG.isna(), -1, 1)})

    df_dat = pd.concat([df_num, df_dat, df_ohc, df_bin], axis = 1)
    with open('df_dat.pkl', 'wb') as p:
        pickle.dump(df_dat, p)
    return df_dat
    
def one_hot_columns(df_cat, noh = 10, nmn = 3, nmx = 100):
    """
    Convert categorical columns to one-hot arrays
    :param df_asm: Categorical assessment data 
    :param noh: Threshold number of levels to start pooling categories.
    :param nmn: Ignore columns with fewer than this many levels.
    :param nmx: Ignore columns with more than this many levels.
    :return: One hot arrays concat'd into one big array
    """
    ohc = []
    for c in df_cat:
        n = df_cat[c].nunique()
        if n < nmn or n > nmx: continue
        if n < noh:
            ohc.append(pd.get_dummies(df_cat[c], prefix = c + '_O', dummy_na = True))
        else:
            ohc.append(pd.get_dummies(f_order(df_cat[c], noh), prefix = c + '_K', dummy_na = False))
    df_ohc = pd.concat(ohc, axis = 1)
    return df_ohc

def date_columns(df_dat):
    """
    Convert date columns to numerical (year plus fraction)
    :param df_dat: Date assessment data.
    :return: Transformed data
    """
    dd = {}
    for c in df_dat:
        d = df_dat[c].str[6:].astype(float) + df_dat[c].str[:2].astype(float) / 13
        dd[c + '_D'] = n_range(d)
    return pd.DataFrame(dd)
    
def numerical_columns(df_num):
    """
    Rescale and remap nan for numerical columns.
    :param df_num: Numerical assessment data.
    :return: Transformed data
    """
    nd = {}
    for c in df_num:
        nd[c + '_N'] = n_range(df_num[c].astype(float))
    return pd.DataFrame(nd)
    

def f_order(x, k):
    """
    Encode the levels of x to k-levels according to frequency.
    :param x: Initial variable with many levels.
    :param k: Number of levels to produce.
    :return: Output variable with k levels.
    """
    # Target level size
    m = (len(x) // k) + 1
    # Level map {x-level: k-level}
    lmap = {}
    # Accumulator, new label
    acc, lbl = 0, None
    for c in Counter(x).most_common():
        if acc + c[1] > m:
            lbl = 1 if lbl is None else (lbl+1)
            acc = c[1]
        else:
            acc += c[1]
        lmap[c[0]] = lbl
    return np.array([lmap[v] for v in x])

def n_range(x):
    """
    Rescale the numerical series x to [0, 1] via rank, map NaN to -1.
    :param x: Numerical series 
    :return: rank(x) / len(x)  (NaN = -1)
    """
    y = np.argsort(np.argsort(x)) / len(x)
    y[np.isnan(x)] = -1
    return y
