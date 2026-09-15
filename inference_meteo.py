"""
Run inference with a pretrained virtual-node GCRNN (GCRNNVirtual) checkpoint
on the meteo dataset: loads the full graph G (real + held-out virtual
stations) for a given k-fold split, loads the matching checkpoint, and
evaluates it on that split's test set, reporting mean absolute error
overall and on the virtual (genuinely unobserved) stations only.

Checkpoints are expected at:
    models/meteo/l={layers}_k={khops}_f={filters}/kfold={fold}.pt

Pretrained checkpoints (fold 0..8, the default filters=128/khops=2/layers=3
configuration) are published on HuggingFace:
    <PLACEHOLDER: HuggingFace model repo URL>
Download and place them under models/meteo/... matching the layout above —
or train your own with virtual_meteo.py.

    python inference_meteo.py                    # fold 0, default model size
    python inference_meteo.py --fold 3            # fold 3
    python inference_meteo.py --fold 3 --filters 64 --khops 2 --layers 3
    python inference_meteo.py --out results/inference/meteo/fold=3  # also save y/yhat arrays
"""

from data_handling.graph import load_graph
from data_handling.timeseries import load_data_m2m, forecast_type
from data_handling.grid import load_NxN_data
from eval_vis.interence import inference
from virtual_meteo import ids_virtual_fromMetadata

import argparse
import os
import sys
import numpy as np
import torch
import torch_geometric_temporal

FEATURES = ['povp. T', 'povp. rel. vla.', 'količina padavin', 'daytime_sin', 'daytime_cos', 'yearday_sin', 'yearday_cos']
FORECAST_FEATURES = ['povp. T', 'povp. rel. vla.']
KNOWN_FEATURES = ['daytime_sin', 'daytime_cos', 'yearday_sin', 'yearday_cos']

HORIZON = 12
HISTORY_WINDOW = 47
BATCH_SIZE = 64
SPLIT = 0.666
res = 240

N_FOLDS = 9  # meteo k-fold data is fixed at 9 folds (0..8)

MODELS_ROOT = 'models/meteo'
HF_MODELS_URL = "<PLACEHOLDER: HuggingFace model repo URL>"


def load_pretrained_model(fold, filters, khops, layers, ctx_features):
    path = f'{MODELS_ROOT}/l={layers}_k={khops}_f={filters}/kfold={fold}.pt'
    if not os.path.isfile(path):
        sys.exit(
            f"error: no checkpoint at {path}\n"
            f"  fold={fold}, filters={filters}, khops={khops}, layers={layers} isn't available locally.\n"
            f"  Download pretrained models from HuggingFace ({HF_MODELS_URL}) and place them\n"
            f"  under {MODELS_ROOT}/... matching that layout, or train your own with virtual_meteo.py."
        )
    from models_impl.GConvGRU import GCRNNVirtual
    model = GCRNNVirtual(filters=filters, layers=layers, khops=khops,
                          node_features=len(FEATURES), forecast_features=len(FORECAST_FEATURES),
                          known_features=len(KNOWN_FEATURES), ctx_features=ctx_features,
                          history=HISTORY_WINDOW, horizon=HORIZON)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Inference with a pretrained virtual-node GCRNN (meteo).')
    parser.add_argument('--fold', type=int, default=0, help='k-fold split to evaluate on (default: %(default)s)')
    parser.add_argument('--filters', type=int, default=128, help='hidden width of the checkpoint to load (default: %(default)s)')
    parser.add_argument('--khops', type=int, default=2, help='Chebyshev filter order of the checkpoint to load (default: %(default)s)')
    parser.add_argument('--layers', type=int, default=3, help='number of GCRNN layers of the checkpoint to load (default: %(default)s)')
    parser.add_argument('--out', default=None, help='directory to save y/yhat/yv/yhatv .npy arrays (default: print summary only)')
    args = parser.parse_args()

    if not (0 <= args.fold < N_FOLDS):
        sys.exit(f"error: --fold must be in [0, {N_FOLDS - 1}]")

    graph_full = 'stations'
    metadata_full = 'data/meteo/contextual/stations.json'
    metadata_virtual = f'data/meteo/contextual/{args.fold}_virtual.json'
    ids_virtual = ids_virtual_fromMetadata(metadata_virtual)

    edge_weight, edge_index = load_graph(graph_full)
    features, targets, mean, std = load_data_m2m(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_full)
    ctx = torch.tensor(load_NxN_data(13, resolution_m=res, metadata=metadata_full), dtype=torch.float32)
    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    _, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    model = load_pretrained_model(args.fold, args.filters, args.khops, args.layers, ctx.shape[-1])

    y, yhat = inference(model, test_dataset, ctx, BATCH_SIZE, ids_virtual)

    y_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in y]
    yhat_np = [t.detach().cpu().numpy().reshape(BATCH_SIZE, targets.shape[1], targets.shape[2], targets.shape[3]) for t in yhat]
    y_np = np.concatenate(y_np)[:, :, -HORIZON:]
    yhat_np = np.concatenate(yhat_np)[:, :, -HORIZON:]
    y_np_n = y_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]
    yhat_np_n = yhat_np * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]
    y_np_n_virtual = y_np_n[:, ids_virtual]
    yhat_np_n_virtual = yhat_np_n[:, ids_virtual]

    summary = np.mean(np.abs(y_np_n - yhat_np_n), axis=(0, 1, 2))
    summary_v = np.mean(np.abs(y_np_n_virtual - yhat_np_n_virtual), axis=(0, 1, 2))

    print(f'fold={args.fold} l={args.layers} k={args.khops} f={args.filters}')
    for name, mae in zip(FORECAST_FEATURES, summary):
        print(f'  {name:20s} MAE (all stations)     = {mae:.4f}')
    for name, mae in zip(FORECAST_FEATURES, summary_v):
        print(f'  {name:20s} MAE (virtual stations) = {mae:.4f}')

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, 'y.npy'), y_np_n)
        np.save(os.path.join(args.out, 'yhat.npy'), yhat_np_n)
        np.save(os.path.join(args.out, 'yv.npy'), y_np_n_virtual)
        np.save(os.path.join(args.out, 'yhatv.npy'), yhat_np_n_virtual)
        print(f'saved y/yhat/yv/yhatv arrays -> {args.out}')


if __name__ == '__main__':
    main()
