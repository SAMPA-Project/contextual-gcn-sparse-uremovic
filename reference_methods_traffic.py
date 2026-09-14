"""
Classical spatial-interpolation baselines (IDW, graph-road-distance variant)
compared against the base GCRNN, on the traffic dataset, across k-fold
splits where a subset of counters is held out as "virtual" (unobserved) nodes.

Run with no arguments to run every fold (0..k-1, k=10 by default) through
both baselines (base_interp, interp_base). Pass --fold to run only that one
fold; pass --k to change the total number of folds (only meaningful if
matching k-fold data was generated with that k).

    python reference_methods_traffic.py             # all folds
    python reference_methods_traffic.py --fold 3     # just fold 3
"""

from data_handling.timeseries import load_data_m2m_traffic
from data_handling.graph import load_graph_traffic
import argparse
import os
import numpy as np

from data_handling.timeseries import forecast_type

FEATURES = [
    'slow', 'fast',
    'VAvg',# 'VMin', 'VMax',
    #'Gap', 'Occ', 'Status',
    'yearday_sin', 'yearday_cos', 'weektime_sin', 'weektime_cos', 'daytime_sin', 'daytime_cos',
    'praznik_Slovenia', 'praznik_Croatia', 'praznik_Germany', 'praznik_Austria', 'praznik_Switzerland', 'praznik_Italy', 'praznik_Hungary',
    'povp. T', 'kolicina padavin',# 'povp. rel. vla.',
    'event_Roadworks'#, 'event_Unknown', 'event_Congestion', 'event_Accident', 'event_Animals', 'event_Vehicle breakdown', 'event_Jam', 'event_Exceptional event', 'event_Driving in the opposite direction', 'event_Special transport', 'event_Other', 'event_Weather event - snow', 'event_Weather event - fog', 'event_Delay at border', 'event_Weather event - wind', 'event_Weather event - storm', 'event_Weather event - hail'
]
FORECAST_FEATURES = ['slow', 'fast', 'VAvg']
KNOWN_FEATURES = [
    'yearday_sin', 'yearday_cos', 'weektime_sin', 'weektime_cos', 'daytime_sin', 'daytime_cos',
    'praznik_Slovenia', 'praznik_Croatia', 'praznik_Germany', 'praznik_Austria', 'praznik_Switzerland', 'praznik_Italy', 'praznik_Hungary',
    'event_Roadworks'#, 'event_Weather event - snow', 'event_Weather event - fog', 'event_Weather event - wind', 'event_Weather event - storm', 'event_Weather event - hail'
]
METADATA_FEATURES = [
    #'onehot_pas1', 'onehot_pas2', 'onehot_pas3',
    'onehot_smer1', 'onehot_smer2',
    'speed_limit',
    'onehot_AC', 'onehot_HC'
]

HORIZON = 6
HISTORY_WINDOW = 23
BATCH_SIZE=64
lr=0.005

SPLIT = 0.666
EPOCHS = 500

def base_interp(k, graph_real, metadata_real, graph_full, metadata_full):
    """
    Baseline: predict on the real (observed-only) graph G-, then spatially
    interpolate (graph-road-distance IDW) those predictions out to the
    virtual (held-out) nodes of the full graph G. Saves per-method errors.

    Args:
        k:             fold index, used only for naming the results folder
        graph_real:    edgelist path for G- (real counters only)
        metadata_real: counter metadata JSON for G-
        graph_full:    edgelist path for G (all counters)
        metadata_full: counter metadata JSON for G
    """
    filters = 64
    layers= 2
    khops = 2
    from hyperparams_traffic import base_model
    y, yhat, _, _ = base_model(graph_real, metadata_real, filters=filters, khops=khops, layers=layers, epochs=EPOCHS, save=False) # predictions on G-
    y_full, _,edge_weight, edge_index = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=1, save=False, skiptrain=True) # labels for G
    import json
    from natsort import natsorted

    with open(metadata_full, 'r') as fp:
        metadataf = json.load(fp)
    with open(metadata_real, 'r') as fp:
        metadatar = json.load(fp)
    full_keys_sorted = natsorted(metadataf.keys())
    real_keys_set    = set(metadatar.keys())
    metaf_reindex = {
        i: {'x': metadataf[k]['x'], 'y': metadataf[k]['y']}
        for i, k in enumerate(full_keys_sorted)
    }
    metar_reindex = {
        i: {'x': metadataf[k]['x'], 'y': metadataf[k]['y']}
        for i, k in enumerate(full_keys_sorted)
        if k in real_keys_set
    }
    all_ids          = list(range(len(full_keys_sorted)))
    real_ids_set     = set(metar_reindex.keys())

    indices_virtual  = [i for i in all_ids if i not in real_ids_set]
    indices_real     = [i for i in all_ids if i in real_ids_set]

    from data_handling.interp_graph import idw_graph as idw
    yhat_idw = idw(yhat,metar_reindex,metaf_reindex,edge_weight,edge_index,power=2,max_hops=2)
    
    real_error = np.mean(np.abs(y - yhat),axis=(0, 1, 2)) # errors on predicted/static
    idw_error = np.mean(np.abs(y_full[:,indices_virtual] - yhat_idw[:,indices_virtual]),axis=(0, 1, 2)) # error on interp idw

    name = f'base_interp_traffic/l={layers}_k={khops}_f={filters}/{k}'
    save_dir = f'results/experiments/{name}'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'y_real.npy'), y)
    np.save(os.path.join(save_dir, 'yhat_real.npy'), yhat)
    np.save(f'{save_dir}/yhat_idw_virtual.npy', yhat_idw[:, indices_virtual])
    np.save(f'{save_dir}/y_full_virtual.npy', y_full[:, indices_virtual])

    np.save(f'{save_dir}/{str(real_error)}_real_error.npy', real_error)
    np.save(f'{save_dir}/{str(idw_error)}_idw_error.npy', idw_error)

def interp_base(k, graph_real, metadata_real, graph_full, metadata_full):
    """
    Baseline: spatially interpolate (graph-road-distance IDW) the raw
    inputs/targets from G- out to all nodes of G first, then run the base
    GCRNN on the full graph using those interpolated inputs. Saves
    per-method errors on both the real and virtual nodes.

    Args: same as base_interp().
    """
    filters = 64
    layers= 2
    khops = 2
    features, targets, mean, std = load_data_m2m_traffic(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_window, metadata_file=metadata_real)
    edge_weight, edge_index = load_graph_traffic(graph_full, metadata_full)
    
    import json
    from natsort import natsorted
    with open(metadata_full, 'r') as fp:
        metadataf = json.load(fp)
    with open(metadata_real, 'r') as fp:
        metadatar = json.load(fp)
    full_keys_sorted = natsorted(metadataf.keys())
    real_keys_set    = set(metadatar.keys())
    metaf_reindex = {
        i: {'x': metadataf[k]['x'], 'y': metadataf[k]['y']}
        for i, k in enumerate(full_keys_sorted)
    }
    metar_reindex = {
        i: {'x': metadataf[k]['x'], 'y': metadataf[k]['y']}
        for i, k in enumerate(full_keys_sorted)
        if k in real_keys_set
    }
    all_ids          = list(range(len(full_keys_sorted)))
    real_ids_set     = set(metar_reindex.keys())
    indices_virtual  = [i for i in all_ids if i not in real_ids_set]
    indices_real     = [i for i in all_ids if i in real_ids_set]

    from data_handling.interp_graph import idw_graph as idw
    targets_idw = idw(targets,metar_reindex,metaf_reindex,edge_weight,edge_index,power=2,max_hops=2)
    features_idw = idw(features,metar_reindex,metaf_reindex,edge_weight,edge_index,power=2,max_hops=2)
    from hyperparams_traffic import base_model
    y, _, _, _ = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=1, save=False, skiptrain=True) # true labels G
    _, yhat_idw, _, _ = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=EPOCHS, save=False, interp_values=(features_idw, targets_idw))

    idw_real_error = np.mean(np.abs(y[:,indices_real] - yhat_idw[:,indices_real]),axis=(0, 1, 2))
    idw_error = np.mean(np.abs(y[:,indices_virtual] - yhat_idw[:,indices_virtual]),axis=(0, 1, 2))

    name = f'interp_base_traffic/l={layers}_k={khops}_f={filters}/{k}'
    save_dir = f'results/experiments/{name}'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'y_real.npy'), y[:,indices_real])
    np.save(f'{save_dir}/y_full_virtual.npy', y[:, indices_virtual])
    np.save(os.path.join(save_dir, 'yhat_idw_real.npy'), yhat_idw[:,indices_real])
    np.save(f'{save_dir}/yhat_idw_virtual.npy', yhat_idw[:, indices_virtual])
    
    np.save(f'{save_dir}/{str(idw_real_error)}_idw_real_error.npy', idw_real_error)
    np.save(f'{save_dir}/{str(idw_error)}_idw_error.npy', idw_error)


def main():
    parser = argparse.ArgumentParser(description='Interpolation baselines vs. base GCRNN (traffic), across k-fold splits.')
    parser.add_argument('--k', type=int, default=10,
                         help='total number of folds the k-fold data was generated with (default: %(default)s)')
    parser.add_argument('--fold', type=int, default=None,
                         help='run only this fold index; default runs every fold 0..k-1')
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else range(args.k)
    for i in folds:
        graph_real = f'data/traffic/graph/{i}_real.nx'
        metadata_real = f"data/traffic/contextual/{i}_real.json"
        metadata_virtual = f"data/traffic/contextual/{i}_virtual.json"
        graph_full = 'data/traffic/graph/stations.nx'
        metadata_full = 'data/traffic/contextual/stations.json'
        base_interp(i, graph_real, metadata_real, graph_full, metadata_full)
        interp_base(i, graph_real, metadata_real, graph_full, metadata_full)

if __name__ == '__main__':
    main()