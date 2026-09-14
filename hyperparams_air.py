"""
Hyperparameter sweep for the base (no virtual nodes) GCRNN on the air
(air-quality station) dataset.

Run with no arguments to sweep every (filters, khops, layers) combination in
the default grid (failures for a given combination are caught and logged,
not fatal). Passing any of --filters/--khops/--layers restricts *that*
dimension to the given single value while the others keep sweeping their
default list.

    python hyperparams_air.py                  # full grid sweep
    python hyperparams_air.py --layers 3         # filters/khops still swept
"""

from data_handling.graph import load_graph_traffic
from data_handling.timeseries import load_data_m2m_air
from data_handling.metadata import load_metadata_features_csv
from training.training import train
from training.training import TrainingMode
from eval_vis.interence import inference
import torch_geometric_temporal

import argparse
import numpy as np
import os
import torch

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
    'settlement_sin', 'settlement_cos', 'settlement_pop', 'settlement_dist_m',
    'settlement_elev_cum', 'settlement_elev_max',
    'lcp_present', 'lcp_sin', 'lcp_cos',
    'lcp_NOx_t_yr', 'lcp_CO_t_yr', 'lcp_PM10_t_yr', 'lcp_CO2_kt_yr',
    'lcp_dist_m', 'lcp_elev_cum', 'lcp_elev_max',
    'ndvi_mean', 'building_coverage'
]
CONTEXTUAL_FEATURES_CSV = "data/air/contextual/contextual_features_normalized.csv"

HORIZON = 6
HISTORY_WINDOW = 23
BATCH_SIZE=128
lr=0.001

SPLIT = 0.666
EPOCHS = 1000
REPORT_TRAIN_LOSS_EPOCHS = 3000

def base_model(graph_real, metadata_real, filters = 64, khops=2, layers=1,epochs=None,save=True,skiptrain=False,interp_values=None):
    """
    Train (or evaluate) the base GCRNN (GCRNNBase) on the air dataset.

    Args:
        graph_real:    path to the edgelist graph file
        metadata_real: path to the station metadata JSON matching that graph
        filters:       hidden channel width
        khops:         Chebyshev filter order (graph conv receptive field)
        layers:        number of stacked GCRNN layers
        epochs:        overrides the module-level EPOCHS for this call if not None
        save:          if True, persist predictions/summary under results/experiments/<name>
        skiptrain:     if True, skip training and only run inference (e.g. to get ground truth)
        interp_values: optional (features, targets) pair to train on instead of the loaded data
                       (used by reference_methods_air.py to feed interpolated inputs)

    Returns:
        (y, yhat, edge_weight, edge_index) — denormalized ground truth/predictions,
        shape (S, N, H, F_out), plus the graph used (handy for reuse by callers).
    """
    if epochs is None:
        epochs = EPOCHS
    edge_weight, edge_index = load_graph_traffic(graph_real, metadata_real)
    features, targets, mean, std = load_data_m2m_air(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_window, metadata_file=metadata_real)
    if interp_values is not None:
        features, targets=interp_values
    ctx = torch.tensor(load_metadata_features_csv(metadata_real, CONTEXTUAL_FEATURES_CSV, METADATA_FEATURES), dtype=torch.float32)

    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    from models_impl.GConvGRU import GCRNNBase
    model = GCRNNBase(filters, layers, filters, len(FEATURES), len(FORECAST_FEATURES), len(KNOWN_FEATURES), ctx.shape[-1], HISTORY_WINDOW, HORIZON, khops)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.HuberLoss(reduction='none')

    name = f'air_aux/l={layers}_k={khops}_f={filters}'

    if not skiptrain:
        train(name, model, train_dataset, test_dataset, loss, optimizer, epochs, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, TrainingMode.MULTIVARIATE_1HORIZON, mean[:len(FORECAST_FEATURES)], std[:len(FORECAST_FEATURES)], station_context=ctx, dynamic_lr=False, BATCH_SIZE_TEST=BATCH_SIZE)
    y, yhat = inference(model, test_dataset, ctx, BATCH_SIZE)

    y_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in y]
    yhat_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in yhat]
    y_np = np.concatenate(y_np)[:,:,-HORIZON:]
    yhat_np = np.concatenate(yhat_np)[:,:,-HORIZON:]
    y_np_n = y_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]
    yhat_np_n = yhat_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]

    if not save:
        return y_np_n, yhat_np_n, edge_weight, edge_index
    save_dir = f'results/experiments/{name}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'y.npy'), y_np_n)
    np.save(os.path.join(save_dir, 'yhat.npy'), yhat_np_n)

    print(f'Saved to: {save_dir}')
    print('y shape:', y_np_n.shape)
    print('yhat shape:', yhat_np_n.shape)
    
    summary = np.mean(np.abs(y_np_n - yhat_np_n),axis=(0, 1, 2))
    np.save(f'{save_dir}/{str(summary)}.npy',np.mean(np.abs(y_np_n - yhat_np_n),axis=(0, 1, 2)))

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for the base GCRNN (air).')
    parser.add_argument('--graph-real', default='data/air/graph/stations_inv.nx',
                         help='edgelist graph path (default: %(default)s)')
    parser.add_argument('--metadata-real', default='data/air/contextual/stations.json',
                         help='station metadata JSON path (default: %(default)s)')
    parser.add_argument('--filters', type=int, default=None,
                         help='hidden width; default sweeps [32, 64, 128]')
    parser.add_argument('--khops', type=int, default=None,
                         help='Chebyshev filter order; default sweeps [2]')
    parser.add_argument('--layers', type=int, default=None,
                         help='number of GCRNN layers; default sweeps [3, 2]')
    args = parser.parse_args()

    filterss = [args.filters] if args.filters is not None else [32, 64, 128]
    khopss   = [args.khops]   if args.khops   is not None else [2]
    layerss  = [args.layers]  if args.layers  is not None else [3, 2]

    for filters in filterss:
        for khops in khopss:
            for layers in layerss:
                try:
                    base_model(args.graph_real, args.metadata_real, filters=filters, khops=khops, layers=layers)
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    main()