"""
Virtual-node GCRNN (GCRNNVirtual) on the meteo dataset: trains on the real
(observed-only) graph G- with a fraction of its nodes masked as "virtual"
during training, then evaluates on the full graph G, forecasting the
genuinely unobserved (virtual) stations.

Run with no arguments to reproduce the original configuration exactly
(filters=128, khops=2, layers=3, virtual_ratio=0.33, context-conditioned
variant, every fold 0..k). Any provided argument overrides just that value.

    python virtual_meteo.py                         # original config, all folds
    python virtual_meteo.py --fold 2 --filters 64     # fold 2 only, filters=64
    python virtual_meteo.py --variant noctx           # raster-context ablation
"""

from data_handling.graph import load_graph
from data_handling.timeseries import load_data_m2m
from data_handling.grid import load_NxN_data
from training.training import train
from training.training import TrainingMode

import argparse
import torch_geometric_temporal
import torch
import os
import numpy as np

from data_handling.timeseries import forecast_type

FEATURES = ['povp. T', 'povp. rel. vla.', 'količina padavin', 'daytime_sin', 'daytime_cos', 'yearday_sin', 'yearday_cos']
FORECAST_FEATURES = ['povp. T', 'povp. rel. vla.']
KNOWN_FEATURES = ['daytime_sin', 'daytime_cos', 'yearday_sin', 'yearday_cos']

HORIZON = 12
HISTORY_WINDOW = 47
BATCH_SIZE=64
lr=0.001

SPLIT = 0.666
EPOCHS = 1000
REPORT_TRAIN_LOSS_EPOCHS = 1000

res = 240

def virtual(layers, khops, filters, virtual_ratio, k, ids_virtual, graph_real, metadata_real, graph_full, metadata_full):
    """
    Train GCRNNVirtual (raster-context-conditioned) on G- with virtual-node
    masking, then evaluate on the full graph G at the true virtual stations.

    Args:
        layers, khops, filters: model hyperparameters
        virtual_ratio:  fraction of G- nodes randomly masked as virtual per
                        training batch (regularizes for the real virtual-node task)
        k:              fold index, used only for naming the results folder
        ids_virtual:    indices (into G's node ordering) of the true held-out stations
        graph_real:     graph variant name for G- (real stations only)
        metadata_real:  station metadata JSON for G-
        graph_full:     graph variant name for G (all stations)
        metadata_full:  station metadata JSON for G
    """
    edge_weight, edge_index = load_graph(graph_real)
    features, targets, mean, std = load_data_m2m(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_real)
    ctx = torch.tensor(load_NxN_data(13, resolution_m=res, metadata=metadata_real), dtype=torch.float32)

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
    loss = torch.nn.L1Loss(reduction='none')

    name = f'vn_residual_1000_horizonfix/vr={virtual_ratio}/l={layers}_k={khops}_f={filters}_lr={lr}/{k}'

    #from training.training import load
    #load(name, model)

    train(name, model, train_dataset, test_dataset, loss, optimizer, EPOCHS, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, TrainingMode.MULTIVARIATE_1HORIZON, mean[:len(FORECAST_FEATURES)], std[:len(FORECAST_FEATURES)], station_context=ctx, dynamic_lr=False, BATCH_SIZE_TEST=BATCH_SIZE, virtual_node_training=True, virtual_node_ratio=virtual_ratio)
    
    edge_weight, edge_index = load_graph(graph_full)
    features, targets, mean, std = load_data_m2m(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_full)
    ctx = torch.tensor(load_NxN_data(13, resolution_m=res, metadata=metadata_full), dtype=torch.float32)
    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    from eval_vis.interence import inference
    y, yhat = inference(model, test_dataset, ctx, BATCH_SIZE, ids_virtual)

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
    Same as virtual(), but the raster context is zeroed out — an ablation
    isolating how much the contextual raster contributes vs. the graph/
    timeseries signal alone. Args: same as virtual().
    """
    edge_weight, edge_index = load_graph(graph_real)
    features, targets, mean, std = load_data_m2m(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_real)
    ctx = torch.tensor(load_NxN_data(13, resolution_m=res, metadata=metadata_real), dtype=torch.float32)
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
    loss = torch.nn.L1Loss(reduction='none')

    name = f'vn_noctx/vr={virtual_ratio}/l={layers}_k={khops}_f={filters}_lr={lr}/{k}'

    #from training.training import load
    #load(name, model)

    train(name, model, train_dataset, test_dataset, loss, optimizer, EPOCHS, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, TrainingMode.MULTIVARIATE_1HORIZON, mean[:len(FORECAST_FEATURES)], std[:len(FORECAST_FEATURES)], station_context=ctx, dynamic_lr=False, BATCH_SIZE_TEST=BATCH_SIZE, virtual_node_training=True, virtual_node_ratio=virtual_ratio)
    
    edge_weight, edge_index = load_graph(graph_full)
    features, targets, mean, std = load_data_m2m(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_full)
    ctx = torch.tensor(load_NxN_data(13, resolution_m=res, metadata=metadata_full), dtype=torch.float32)
    ctx = torch.zeros_like(ctx)
    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    from eval_vis.interence import inference
    y, yhat = inference(model, test_dataset, ctx, BATCH_SIZE, ids_virtual)

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

def ids_virtual_fromMetadata(metadata_virtual):
    """Read the held-out-station metadata JSON and return their node indices in G."""
    with open(metadata_virtual, 'r') as fp:
        import json
        m = json.load(fp)
        ids = []
        for item in m:
            ids.append(m[item]['idx'])
    return ids

def main():
    parser = argparse.ArgumentParser(description='Virtual-node GCRNN (meteo), across k-fold splits.')
    parser.add_argument('--k', type=int, default=8,
                         help='total number of folds the k-fold data was generated with (default: %(default)s)')
    parser.add_argument('--fold', type=int, default=None,
                         help='run only this fold index; default runs every fold 0..k')
    parser.add_argument('--filters', type=int, default=128, help='hidden width (default: %(default)s)')
    parser.add_argument('--khops', type=int, default=2, help='Chebyshev filter order (default: %(default)s)')
    parser.add_argument('--layers', type=int, default=3, help='number of GCRNN layers (default: %(default)s)')
    parser.add_argument('--virtual-ratio', type=float, default=0.33,
                         help='fraction of G- nodes masked as virtual per training batch (default: %(default)s)')
    parser.add_argument('--variant', choices=['ctx', 'noctx'], default='ctx',
                         help="'ctx' conditions on the raster context, 'noctx' zeroes it out (ablation) (default: %(default)s)")
    args = parser.parse_args()

    run = virtual if args.variant == 'ctx' else virtual_noctx
    k = args.k
    folds = [args.fold] if args.fold is not None else range(k + 1)
    for i in folds:
        graph_real = f'{i}_real'
        metadata_real = f'data/meteo/contextual/{i}_real.json'
        metadata_virtual = f'data/meteo/contextual/{i}_virtual.json'
        graph_full = 'stations' # this is the graph that was virtualized fold-by-fold
        metadata_full = 'data/meteo/contextual/stations.json'
        ids_virtual = ids_virtual_fromMetadata(metadata_virtual)
        run(args.layers, args.khops, args.filters, args.virtual_ratio, i, ids_virtual, graph_real, metadata_real, graph_full, metadata_full)

if __name__ == '__main__':
    main()