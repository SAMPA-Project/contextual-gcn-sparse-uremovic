import glob
import numpy as np
import pandas as pd
from natsort import natsorted

def normalize(features, mean=None, std=None):
    ax = (0, 1) if features.ndim == 3 else 0
    if mean is None:
        mean = np.mean(features, axis=ax)
    if std is None:
        std = np.std(features, axis=ax)
    return ((features - mean) / std, mean, std)

def denormalize(features, mean, std):
    return features * std + mean

from enum import Enum
class forecast_type(Enum):
    horizon_window=1
    horizon_only_window=2
    hrzowd_sparse=3

# traffic
def load_data_m2m_traffic(history_window, horizon, features, forecast_features, ft, metadata_file):
    ts_files = natsorted(glob.glob('data/traffic/timeseries/*.csv'))
    dfs = []
    with open(metadata_file, 'r') as fp:
        import json
        mtdt = '\n'.join(fp.readlines())
        metadata = json.loads(mtdt)
    for fn in ts_files:
        id = fn[-10:-4] # :-4 cuts .csv, -10: cuts prefix path, remains id of counter (filename no extension) # consolidated
        if id in metadata.keys():
            dfs.append(pd.read_csv(fn).drop(range(40692,40884)).drop(range(47772,48732)).drop(range(51300,51468)).drop(range(57612,57732))[-365*3*24:])

    return lm2m(dfs, history_window, horizon, features, forecast_features, ft, metadata_file)

# air
def load_data_m2m_air(history_window, horizon, features, forecast_features, ft, metadata_file):
    ts_files = natsorted(glob.glob('data/air/timeseries/*.csv'))
    dfs = []
    with open(metadata_file, 'r') as fp:
        import json
        mtdt = '\n'.join(fp.readlines())
        metadata = json.loads(mtdt)
    for fn in ts_files:
        id = fn[-8:-4] # :-4 cuts .csv, -8: cuts prefix path, remains station id (filename no extension)
        if id in metadata.keys():
            dfs.append(pd.read_csv(fn))

    return lm2m(dfs, history_window, horizon, features, forecast_features, ft, metadata_file)

# meteo
def load_data_m2m(history_window, horizon, features, forecast_features, demo_nsamples=None, ft=forecast_type.horizon_window, metadata_file='data/meteo/contextual/stations.json'):
    ts_files = natsorted(glob.glob('data/meteo/timeseries/**'))
    dfs = []
    with open(metadata_file, 'r') as fp:
        import json
        mtdt = '\n'.join(fp.readlines())
        metadata = json.loads(mtdt)
    for fn in ts_files:
        id = 'id_' + fn[-8:-4]
        if id in metadata.keys():
            dfs.append(pd.read_csv(fn))
    return lm2m(dfs, history_window, horizon, features, forecast_features, ft, metadata_file, demo_nsamples)

# base
def lm2m(dfs, history_window, horizon, features, forecast_features, ft, metadata_file, demo_nsamples=None):
    if demo_nsamples:
        n_samples = demo_nsamples
    else:
        n_samples = len(dfs[0])
    n_nodes = len(dfs)
    n_features = len(features)
    n_fc_features = len(forecast_features)

    features_ = np.zeros(shape=(n_samples, n_nodes, n_features))
    targets_ = np.zeros(shape=(n_samples, n_nodes, n_fc_features))
    for i, df in enumerate(dfs):
        features_[:,i] = np.array(df[features]) if not demo_nsamples else np.array(df[features])[:demo_nsamples]
        targets_[:,i] = np.array(df[forecast_features]) if not demo_nsamples else np.array(df[forecast_features])[:demo_nsamples]

    # fill nan with linear interp
    mask = np.isnan(features_)
    for i in range(n_nodes):
        for j in range(n_features):
            mask_part = mask[:,i][:,j]
            if len(features_[:,i][:,j][~mask_part]) > 0:
                features_[:,i][:,j][mask_part] = np.interp(np.flatnonzero(mask_part), np.flatnonzero(~mask_part), features_[:,i][:,j][~mask_part])
            else:
                features_[:,i][:,j][mask_part] = np.zeros(shape=features_[:,i][:,j][mask_part].shape)
        for j in range(n_fc_features):
            mask_part = mask[:,i][:,j]
            if len(targets_[:,i][:,j][~mask_part]) > 0:
                targets_[:,i][:,j][mask_part] = np.interp(np.flatnonzero(mask_part), np.flatnonzero(~mask_part), targets_[:,i][:,j][~mask_part])
            else:
                targets_[:,i][:,j][mask_part] = np.zeros(shape=targets_[:,i][:,j][mask_part].shape)

    features_, mean, std = normalize(features_)
    targets_, _, _ = normalize(targets_, mean[:targets_.shape[-1]], std[:targets_.shape[-1]])

    if ft==forecast_type.horizon_window:
        lags = history_window
        lags_horizon = horizon
        features = np.zeros(shape=(n_samples - lags - 1, n_nodes, lags + lags_horizon, n_features))
        targets = np.zeros(shape=(n_samples - lags - 1, n_nodes, lags + lags_horizon, n_fc_features))
        for i in range(lags, n_samples - lags_horizon - 1):
            features[i - lags] = np.swapaxes(features_[i-lags:i+lags_horizon], axis1=1, axis2=0)
            targets[i - lags] = np.swapaxes(targets_[i-lags+1:i+lags_horizon+1], axis1=1, axis2=0)
    elif ft==forecast_type.horizon_only_window:
        lags = history_window
        lags_horizon = horizon
        features = np.zeros(shape=(n_samples - lags - 1, n_nodes, lags + lags_horizon, n_features))
        targets = np.zeros(shape=(n_samples - lags - 1, n_nodes, lags_horizon, n_fc_features))
        for i in range(lags, n_samples - lags_horizon - 1):
            features[i - lags] = np.swapaxes(features_[i-lags:i+lags_horizon], axis1=1, axis2=0)
            targets[i - lags] = np.swapaxes(targets_[i+1:i+lags_horizon+1], axis1=1, axis2=0)
    elif ft==forecast_type.hrzowd_sparse: # save space
        lags = history_window
        lags_horizon = horizon
        sparseness = 59
        features = np.zeros(shape=((n_samples - lags - 1)//sparseness+1, n_nodes, lags + lags_horizon, n_features))
        targets = np.zeros(shape=((n_samples - lags - 1)//sparseness+1, n_nodes, lags_horizon, n_fc_features))
        cnt = 0
        for i in range(lags, n_samples - lags_horizon - 1, sparseness):
            features[cnt] = np.swapaxes(features_[i-lags:i+lags_horizon], axis1=1, axis2=0)
            targets[cnt] = np.swapaxes(targets_[i:i+lags_horizon], axis1=1, axis2=0)
            cnt+=1

    return (features, targets, mean, std)
