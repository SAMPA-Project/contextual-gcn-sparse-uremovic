"""
Virtual-node GCRNN (GCRNNVirtual) on the traffic dataset: trains on the real
(observed-only) graph G- with a fraction of its nodes masked as "virtual"
during training, then evaluates on the full graph G, forecasting the
genuinely unobserved (virtual) counters.

Run with no arguments to reproduce the original configuration exactly
(filters=192, khops=2, layers=3, virtual_ratio=0.33, context-free ablation
variant, every fold 0..k-1). Any provided argument overrides just that value.

    python virtual_traffic.py                         # original config, all folds
    python virtual_traffic.py --fold 2 --filters 64     # fold 2 only, filters=64
    python virtual_traffic.py --variant ctx             # context-conditioned variant
"""

from data_handling.graph import load_graph_traffic
from data_handling.timeseries import load_data_m2m_traffic
from data_handling.metadata import load_metadata_features
from training.training import train
from training.training import TrainingMode

import argparse
import torch_geometric_temporal
import torch
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
lr=0.001

SPLIT = 0.666
EPOCHS = 1000
REPORT_TRAIN_LOSS_EPOCHS = 3000

res = 240

def virtual(layers, khops, filters, virtual_ratio, k, ids_virtual, graph_real, metadata_real, graph_full, metadata_full):
    """
    Train GCRNNVirtual (context-conditioned) on G- with virtual-node
    masking and reconnected paths, then evaluate on the full graph G at the
    true virtual counters.

    Args:
        layers, khops, filters: model hyperparameters
        virtual_ratio:  fraction of G- nodes randomly masked as virtual per
                        training batch (regularizes for the real virtual-node task)
        k:              fold index, used only for naming the results folder
        ids_virtual:    indices (into G's node ordering) of the true held-out counters
        graph_real:     edgelist path for G- (real counters only)
        metadata_real:  counter metadata JSON for G-
        graph_full:     edgelist path for G (all counters)
        metadata_full:  counter metadata JSON for G
    """
    edge_weight, edge_index = load_graph_traffic(graph_real,metadata_real)
    features, targets, mean, std = load_data_m2m_traffic(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_real)
    ctx = torch.tensor(load_metadata_features(metadata_real, METADATA_FEATURES), dtype=torch.float32)

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

    name = f'vn_traffic/vr={virtual_ratio}/l={layers}_k={khops}_f={filters}_lr={lr}/{k}'

    #from training.training import load
    #load(name, model)

    train(name, model, train_dataset, test_dataset, loss, optimizer, EPOCHS, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, TrainingMode.MULTIVARIATE_1HORIZON, mean[:len(FORECAST_FEATURES)], std[:len(FORECAST_FEATURES)], station_context=ctx, dynamic_lr=False, BATCH_SIZE_TEST=BATCH_SIZE, virtual_node_training=True, virtual_node_ratio=virtual_ratio, reconnect_paths=True)
    
    edge_weight, edge_index = load_graph_traffic(graph_full, metadata_full)
    features, targets, mean, std = load_data_m2m_traffic(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_full)
    ctx = torch.tensor(load_metadata_features(metadata_full, METADATA_FEATURES), dtype=torch.float32)
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

def virtual_noctx(layers, khops, filters, virtual_ratio, k, ids_virtual, graph_real, metadata_real, graph_full, metadata_full):
    """
    Same as virtual(), but the context vector is zeroed out — an ablation
    isolating how much the contextual features contribute vs. the graph/
    timeseries signal alone. Args: same as virtual().
    """
    edge_weight, edge_index = load_graph_traffic(graph_real,metadata_real)
    features, targets, mean, std = load_data_m2m_traffic(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_real)
    ctx = torch.tensor(load_metadata_features(metadata_real, METADATA_FEATURES), dtype=torch.float32)
    ctx = torch.zeros_like(ctx)

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

    name = f'vn_traffic_noctx/vr={virtual_ratio}/l={layers}_k={khops}_f={filters}_lr={lr}/{k}'

    #from training.training import load
    #load(name, model)

    train(name, model, train_dataset, test_dataset, loss, optimizer, EPOCHS, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, TrainingMode.MULTIVARIATE_1HORIZON, mean[:len(FORECAST_FEATURES)], std[:len(FORECAST_FEATURES)], station_context=ctx, dynamic_lr=False, BATCH_SIZE_TEST=BATCH_SIZE, virtual_node_training=True, virtual_node_ratio=virtual_ratio, reconnect_paths=True)
    
    edge_weight, edge_index = load_graph_traffic(graph_full, metadata_full)
    features, targets, mean, std = load_data_m2m_traffic(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_full)
    ctx = torch.tensor(load_metadata_features(metadata_full, METADATA_FEATURES), dtype=torch.float32)
    ctx = torch.zeros_like(ctx)
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
    """Read the held-out-counter metadata JSON and return their node indices in G."""
    from natsort import natsorted
    import json

    with open(metadata_virtual, 'r') as f:
        meta_v = json.load(f)
    with open(metadata_full, 'r') as f:
        meta_f = json.load(f)

    full_keys_sorted = natsorted(meta_f.keys())
    key_to_idx = {k: i for i, k in enumerate(full_keys_sorted)}

    return sorted([key_to_idx[k] for k in meta_v.keys()])

def main():
    parser = argparse.ArgumentParser(description='Virtual-node GCRNN (traffic), across k-fold splits.')
    parser.add_argument('--k', type=int, default=10,
                         help='total number of folds the k-fold data was generated with (default: %(default)s)')
    parser.add_argument('--fold', type=int, default=None,
                         help='run only this fold index; default runs every fold 0..k-1')
    parser.add_argument('--filters', type=int, default=192, help='hidden width (default: %(default)s)')
    parser.add_argument('--khops', type=int, default=2, help='Chebyshev filter order (default: %(default)s)')
    parser.add_argument('--layers', type=int, default=3, help='number of GCRNN layers (default: %(default)s)')
    parser.add_argument('--virtual-ratio', type=float, default=0.33,
                         help='fraction of G- nodes masked as virtual per training batch (default: %(default)s)')
    parser.add_argument('--variant', choices=['ctx', 'noctx'], default='noctx',
                         help="'ctx' conditions on the contextual features, 'noctx' zeroes them out (ablation) (default: %(default)s)")
    args = parser.parse_args()

    run = virtual if args.variant == 'ctx' else virtual_noctx
    folds = [args.fold] if args.fold is not None else range(args.k)
    for i in folds:
        graph_real = f'data/traffic/graph/{i}_real_expw.nx'
        metadata_real = f"data/traffic/contextual/{i}_real.json"
        metadata_virtual = f"data/traffic/contextual/{i}_virtual.json"
        graph_full = 'data/traffic/graph/stations_expw.nx'
        metadata_full = 'data/traffic/contextual/stations.json'
        ids_virtual = ids_virtual_fromMetadata(metadata_virtual, metadata_full)
        run(args.layers, args.khops, args.filters, args.virtual_ratio, i, ids_virtual, graph_real, metadata_real, graph_full, metadata_full)

if __name__ == '__main__':
    main()