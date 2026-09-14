"""
Hyperparameter sweep for the base (no virtual nodes) GCRNN on the traffic
(motorway counter) dataset.

Run with no arguments to sweep every (filters, khops, layers) combination in
the default grid. Passing any of --filters/--khops/--layers restricts *that*
dimension to the given single value while the others keep sweeping their
default list.

    python hyperparams_traffic.py                 # full grid sweep
    python hyperparams_traffic.py --khops 1        # filters/layers still swept
"""

from data_handling.graph import load_graph_traffic
from data_handling.timeseries import load_data_m2m_traffic
from data_handling.metadata import load_metadata_features
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

def base_model(g, m, filters = 64, khops=2, layers=1, epochs=None, save=True, skiptrain=False, interp_values=None):
    """
    Train (or evaluate) the base GCRNN (GCRNNBase) on the traffic dataset.

    Args:
        g:             path to the edgelist graph file
        m:             path to the counter metadata JSON matching that graph
        filters:       hidden channel width
        khops:         Chebyshev filter order (graph conv receptive field)
        layers:        number of stacked GCRNN layers
        epochs:        overrides the module-level EPOCHS for this call if not None
        save:          if True, persist predictions/summary under results/experiments/<name>
        skiptrain:     if True, skip training and only run inference (e.g. to get ground truth)
        interp_values: optional (features, targets) pair to train on instead of the loaded data
                       (used by reference_methods_traffic.py to feed interpolated inputs)

    Returns:
        (y, yhat, edge_weight, edge_index) — denormalized ground truth/predictions,
        shape (S, N, H, F_out), plus the graph used (handy for reuse by callers).
    """
    if epochs is not None:
        global EPOCHS
        EPOCHS=epochs
    edge_weight, edge_index = load_graph_traffic(g,m)
    features, targets, mean, std = load_data_m2m_traffic(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_window, metadata_file=m)
    if interp_values is not None:
        features, targets=interp_values
    ctx = torch.tensor(load_metadata_features(m, METADATA_FEATURES), dtype=torch.float32)

    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    from models_impl.GConvGRU import GCRNNBase
    model = GCRNNBase(filters, layers, filters, len(FEATURES), len(FORECAST_FEATURES), len(KNOWN_FEATURES), ctx.shape[-1], HISTORY_WINDOW, HORIZON, khops)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.HuberLoss(reduction='none')

    name = f'hyperparams_traffic_hist23_hubertloss_gradientclip/l={layers}_k={khops}_f={filters}'

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
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for the base GCRNN (traffic).')
    parser.add_argument('--graph-real', default='data/traffic/graph/stations.nx',
                         help='edgelist graph path (default: %(default)s)')
    parser.add_argument('--metadata-real', default='data/traffic/contextual/stations.json',
                         help='counter metadata JSON path (default: %(default)s)')
    parser.add_argument('--filters', type=int, default=None,
                         help='hidden width; default sweeps [256]')
    parser.add_argument('--khops', type=int, default=None,
                         help='Chebyshev filter order; default sweeps [1, 2]')
    parser.add_argument('--layers', type=int, default=None,
                         help='number of GCRNN layers; default sweeps [3]')
    args = parser.parse_args()

    filterss = [args.filters] if args.filters is not None else [256]
    khopss   = [args.khops]   if args.khops   is not None else [1, 2]
    layerss  = [args.layers]  if args.layers  is not None else [3]

    for filters in filterss:
        for layers in layerss:
            for khops in khopss:
                base_model(args.graph_real, args.metadata_real, filters=filters, khops=khops, layers=layers)

if __name__ == '__main__':
    main()