"""
Virtual-node GCRNN (GCRNNVirtual) on the air dataset: trains on the real
(observed-only) graph G- with a fraction of its nodes masked as "virtual"
during training, then evaluates on the full graph G, forecasting the
genuinely unobserved (virtual) stations.

Always conditioned on the station context features (this is the proposed
method, not an ablation of it).

Run with no arguments to reproduce the original configuration exactly
(filters=64, khops=2, layers=3, virtual_ratio=0.2, every fold 0..7). Any
provided argument overrides just that value.

    python virtual_air.py                         # original config, all folds
    python virtual_air.py --fold 2 --filters 32     # fold 2 only, filters=32
"""

from data_handling.graph import load_graph_traffic
from data_handling.timeseries import load_data_m2m_air
from data_handling.metadata import load_metadata_features_csv
from training.training import train
from training.training import TrainingMode

import argparse
import torch_geometric_temporal
import torch
import os
import numpy as np

from data_handling.timeseries import forecast_type

FEATURES = [
    'NO2', 'O3', 'PM10',
    'povp. T', 'kolicina padavin', 'povp. rel. vla.', 'hitrost_vetra', 'povp_tlak',
    'yearday_sin', 'yearday_cos', 'weektime_sin', 'weektime_cos', 'daytime_sin', 'daytime_cos'    
]
FORECAST_FEATURES = ['NO2', 'O3', 'PM10']
KNOWN_FEATURES = [
    'yearday_sin', 'yearday_cos', 'weektime_sin', 'weektime_cos', 'daytime_sin', 'daytime_cos'
]
METADATA_FEATURES = [
    # settlement
    'settlement_sin', 'settlement_cos',
    'settlement_pop', 'settlement_dist_m',
    'settlement_elev_cum', 'settlement_elev_max',
    # lcp
    'lcp_present',
    'lcp_sin', 'lcp_cos',
    'lcp_NOx_t_yr', 'lcp_CO_t_yr', 'lcp_PM10_t_yr', 'lcp_CO2_kt_yr',
    'lcp_dist_m', 'lcp_elev_cum', 'lcp_elev_max',
    # raster
    'ndvi_mean', 'building_coverage',
    # timeseries stats (3 pollutants x 5 stats = 15)
    'NO2_mean',  'NO2_std',  'NO2_p5',  'NO2_p95',  'NO2_daily_fluct',
    'O3_mean',   'O3_std',   'O3_p5',   'O3_p95',   'O3_daily_fluct',
    'PM10_mean', 'PM10_std', 'PM10_p5', 'PM10_p95', 'PM10_daily_fluct',
]
CONTEXTUAL_FEATURES_CSV = "data/air/contextual/contextual_features_normalized.csv"

HORIZON = 6
HISTORY_WINDOW = 23
BATCH_SIZE=32
lr=0.001

SPLIT = 0.666
EPOCHS = 1000
REPORT_TRAIN_LOSS_EPOCHS = 3000

def virtual(layers, khops, filters, virtual_ratio, k, ids_virtual, graph_real, metadata_real, graph_full, metadata_full):
    """
    Train GCRNNVirtual (context-conditioned) on G- with virtual-node
    masking and reconnected paths, then evaluate on the full graph G at the
    true virtual stations.

    Args:
        layers, khops, filters: model hyperparameters
        virtual_ratio:  fraction of G- nodes randomly masked as virtual per
                        training batch (regularizes for the real virtual-node task)
        k:              fold index, used only for naming the results folder
        ids_virtual:    indices (into G's node ordering) of the true held-out stations
        graph_real:     edgelist path for G- (real stations only)
        metadata_real:  station metadata JSON for G-
        graph_full:     edgelist path for G (all stations)
        metadata_full:  station metadata JSON for G
    """
    edge_weight, edge_index = load_graph_traffic(graph_real,metadata_real)
    features, targets, mean, std = load_data_m2m_air(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_real)
    ctx = torch.tensor(load_metadata_features_csv(metadata_real, CONTEXTUAL_FEATURES_CSV, METADATA_FEATURES), dtype=torch.float32)

    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    from models_impl.GConvGRU import GCRNNVirtual
    model = GCRNNVirtual(filters=filters,
                                                layers=layers,
                                                khops=khops,
                                                node_features=len(FEATURES),
                                                forecast_features=len(FORECAST_FEATURES),
                                                known_features=len(KNOWN_FEATURES),
                                                ctx_features=ctx.shape[-1],
                                                history=HISTORY_WINDOW,
                                                horizon=HORIZON)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss(reduction='none')

    name = f'vn_air_big/vr={virtual_ratio}/l={layers}_k={khops}_f={filters}_lr={lr}/{k}'

    #from training.training import load
    #load(name, model)

    train(name, model, train_dataset, test_dataset, loss, optimizer, EPOCHS, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, TrainingMode.MULTIVARIATE_1HORIZON, mean[:len(FORECAST_FEATURES)], std[:len(FORECAST_FEATURES)], station_context=ctx, dynamic_lr=False, BATCH_SIZE_TEST=BATCH_SIZE, virtual_node_training=True, virtual_node_ratio=virtual_ratio, reconnect_paths=True)
    
    edge_weight, edge_index = load_graph_traffic(graph_full, metadata_full)
    features, targets, mean, std = load_data_m2m_air(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_full)
    ctx = torch.tensor(load_metadata_features_csv(metadata_full, CONTEXTUAL_FEATURES_CSV, METADATA_FEATURES), dtype=torch.float32)
    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    from eval_vis.interence import inference
    y, yhat = inference(model, test_dataset, ctx, BATCH_SIZE, ids_virtual, reconnect_paths=True)

    save_dir = f'results/experiments/{name}'
    os.makedirs(save_dir, exist_ok=True)

    y_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in y]
    yhat_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in yhat]
    y_np = np.concatenate(y_np)[:,:,-HORIZON:]
    yhat_np = np.concatenate(yhat_np)[:,:,-HORIZON:]
    y_np_n = y_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]
    yhat_np_n = yhat_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]
    y_np_n_virtual = y_np_n[:,ids_virtual]
    yhat_np_n_virtual = yhat_np_n[:,ids_virtual]
    np.save(os.path.join(save_dir, 'y.npy'), y_np_n)
    np.save(os.path.join(save_dir, 'yhat.npy'), yhat_np_n)
    np.save(os.path.join(save_dir, 'yv.npy'), y_np_n_virtual)
    np.save(os.path.join(save_dir, 'yhatv.npy'), yhat_np_n_virtual)
    
    summary = np.mean(np.abs(y_np_n - yhat_np_n),axis=(0, 1, 2))
    np.save(f'{save_dir}/{str(summary)}.npy',summary)
    summary_v = np.mean(np.abs(y_np_n_virtual - yhat_np_n_virtual),axis=(0, 1, 2))
    np.save(f'{save_dir}/{str(summary_v)}_virtual.npy',summary_v)

    return

def ids_virtual_fromMetadata(metadata_virtual, metadata_full):
    """Read the held-out-station metadata JSON and return their node indices in G."""
    from natsort import natsorted
    import json

    with open(metadata_virtual, 'r') as f:
        meta_v = json.load(f)
    with open(metadata_full, 'r') as f:
        meta_f = json.load(f)

    full_keys_sorted = natsorted(meta_f.keys())
    key_to_idx = {k: i for i, k in enumerate(full_keys_sorted)}

    return sorted([key_to_idx[k] for k in meta_v.keys()])

N_FOLDS = 8  # air k-fold data is fixed at 8 folds (0..7)

def main():
    parser = argparse.ArgumentParser(description='Virtual-node GCRNN (air), across k-fold splits.')
    parser.add_argument('--fold', type=int, default=None,
                         help='run only this fold index; default runs every fold 0..7')
    parser.add_argument('--filters', type=int, default=64, help='hidden width (default: %(default)s)')
    parser.add_argument('--khops', type=int, default=2, help='Chebyshev filter order (default: %(default)s)')
    parser.add_argument('--layers', type=int, default=3, help='number of GCRNN layers (default: %(default)s)')
    parser.add_argument('--virtual-ratio', type=float, default=0.2,
                         help='fraction of G- nodes masked as virtual per training batch (default: %(default)s)')
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else range(N_FOLDS)
    for i in folds:
        graph_real = f'data/air/graph/{i}_real_inv.nx'
        metadata_real = f"data/air/contextual/{i}_real.json"
        metadata_virtual = f"data/air/contextual/{i}_virtual.json"
        graph_full = "data/air/graph/stations_inv.nx"
        metadata_full = 'data/air/contextual/stations.json'
        ids_virtual = ids_virtual_fromMetadata(metadata_virtual, metadata_full)
        virtual(args.layers, args.khops, args.filters, args.virtual_ratio, i, ids_virtual, graph_real, metadata_real, graph_full, metadata_full)

if __name__ == '__main__':
    main()