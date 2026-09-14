import json
import numpy as np

def load_metadata_features_csv(metadata_path, csv_path, fs):
    import pandas as pd
    from natsort import natsorted

    # get canonical node order from metadata json
    with open(metadata_path, 'r') as fp:
        metadata = json.load(fp)
    sorted_keys = natsorted(metadata.keys())   # e.g. ['E403', 'E407', ...]

    # load csv, index by node_id
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)

    # reindex to match metadata node order
    df = df.loc[sorted_keys]

    features = df[fs].values.astype(np.float32)  # (N, len(fs))
    features = np.nan_to_num(features, nan=0.0)  # (N, len(fs))
    return features

def load_metadata_features(path, fs):
    with open(path, 'r') as fp:
        metadata = json.load(fp)
    
    myKeys = list(metadata.keys())
    myKeys.sort()
    metadata_sorted = {i: metadata[i] for i in myKeys}
    metadata = metadata_sorted

    features = np.zeros(shape=(len(metadata), len(fs)))
    for i,s in enumerate(metadata.keys()):
        for j,f in enumerate(fs):
            features[i][j] = metadata[s][f]
    
    return features