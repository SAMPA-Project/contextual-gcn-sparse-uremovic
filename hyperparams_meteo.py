"""
Hyperparameter sweep for the base (no virtual nodes) GCRNN on the meteo
(weather station) dataset.

Run with no arguments to sweep every (filters, khops, layers) combination in
the default grid, training and evaluating one model per combination. Passing
any of --filters/--khops/--layers restricts *that* dimension to the given
single value while the others keep sweeping their default list.

    python hyperparams_meteo.py                  # full grid sweep
    python hyperparams_meteo.py --filters 256     # khops/layers still swept
    python hyperparams_meteo.py --filters 256 --khops 2 --layers 3  # single run
"""

from data_handling.graph import load_graph
from data_handling.timeseries import load_data_m2m
from data_handling.grid import load_NxN_data
from training.training import train
from training.training import TrainingMode
from eval_vis.interence import inference
import torch_geometric_temporal

import argparse
import numpy as np
import os
import torch

from data_handling.timeseries import forecast_type

FEATURES = ['povp. T', 'povp. rel. vla.', 'količina padavin', 'daytime_sin', 'daytime_cos', 'yearday_sin', 'yearday_cos']
FORECAST_FEATURES = ['povp. T', 'povp. rel. vla.']
KNOWN_FEATURES = ['daytime_sin', 'daytime_cos', 'yearday_sin', 'yearday_cos']

HORIZON = 12
HISTORY_WINDOW = 47
BATCH_SIZE=64
lr=0.005

SPLIT = 0.666
EPOCHS = 200
REPORT_TRAIN_LOSS_EPOCHS = 1000

res = 240

def base_model(graph_real, metadata_real, filters = 64, khops=2, layers=1, epochs=None, save=True, skiptrain=False, interp_values=None):
    """
    Train (or evaluate) the base GCRNN (GCRNNBase) on the meteo dataset.

    Args:
        graph_real:    graph variant name under data/meteo/graph/<graph_real>.nx
        metadata_real: path to the station metadata JSON for this graph
        filters:       hidden channel width
        khops:         Chebyshev filter order (graph conv receptive field)
        layers:        number of stacked GCRNN layers
        epochs:        overrides the module-level EPOCHS for this call if not None
        save:          if True, persist predictions/summary under results/experiments/<name>
        skiptrain:     if True, skip training and only run inference (e.g. to get ground truth)
        interp_values: optional (features, targets) pair to train on instead of the loaded data
                       (used by reference_methods_meteo.py to feed interpolated inputs)

    Returns:
        (y, yhat) ground truth and prediction arrays, denormalized, shape (S, N, H, F_out).
    """
    if epochs is not None:
        global EPOCHS
        EPOCHS=epochs
    edge_weight, edge_index = load_graph(graph_real)
    features, targets, mean, std = load_data_m2m(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_window, metadata_file=metadata_real)
    if interp_values is not None:
        features, targets=interp_values
    ctx = torch.tensor(load_NxN_data(13, resolution_m=res, metadata=metadata_real), dtype=torch.float32)

    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    from models_impl.GConvGRU import GCRNNBase
    model = GCRNNBase(filters, layers, filters, len(FEATURES), len(FORECAST_FEATURES), len(KNOWN_FEATURES), ctx.shape[-1], HISTORY_WINDOW, HORIZON, khops)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.L1Loss(reduction='none')

    name = f'hyperparams_200_horizononly_/l={layers}_k={khops}_f={filters}'

    if not skiptrain:
        train(name, model, train_dataset, test_dataset, loss, optimizer, EPOCHS, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, TrainingMode.MULTIVARIATE_1HORIZON, mean[:len(FORECAST_FEATURES)], std[:len(FORECAST_FEATURES)], station_context=ctx, dynamic_lr=False, BATCH_SIZE_TEST=BATCH_SIZE)
    y, yhat = inference(model, test_dataset, ctx, BATCH_SIZE)

    y_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in y]
    yhat_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in yhat]
    y_np = np.concatenate(y_np)[:,:,-HORIZON:]
    yhat_np = np.concatenate(yhat_np)[:,:,-HORIZON:]
    y_np_n = y_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]
    yhat_np_n = yhat_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]

    if not save:
        return y_np_n, yhat_np_n
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
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for the base GCRNN (meteo).')
    parser.add_argument('--graph-real', default='stations_hp',
                         help='graph variant name under data/meteo/graph/ (default: %(default)s)')
    parser.add_argument('--metadata-real', default='data/meteo/contextual/stations.json',
                         help='station metadata JSON path (default: %(default)s)')
    parser.add_argument('--filters', type=int, default=None,
                         help='hidden width; default sweeps [128, 192, 256, 320, 384]')
    parser.add_argument('--khops', type=int, default=None,
                         help='Chebyshev filter order; default sweeps [2, 3]')
    parser.add_argument('--layers', type=int, default=None,
                         help='number of GCRNN layers; default sweeps [2, 3]')
    args = parser.parse_args()

    filterss = [args.filters] if args.filters is not None else [128, 192, 256, 320, 384]
    khopss   = [args.khops]   if args.khops   is not None else [2, 3]
    layerss  = [args.layers]  if args.layers  is not None else [2, 3]

    for filters in filterss:
        for layers in layerss:
            for khops in khopss:
                base_model(args.graph_real, args.metadata_real, filters=filters, khops=khops, layers=layers)

if __name__ == '__main__':
    main()