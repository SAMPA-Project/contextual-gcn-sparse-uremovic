"""
Run inference with a pretrained virtual-node GCRNN (GCRNNVirtual) checkpoint
on the traffic dataset: loads the full graph G (real + held-out virtual
counters) for a given k-fold split, loads the matching checkpoint, and
evaluates it on that split's test set, reporting mean absolute error
overall and on the virtual (genuinely unobserved) counters only.

Checkpoints are expected at:
    models/traffic/l={layers}_k={khops}_f={filters}/kfold={fold}.pt

Pretrained checkpoints (fold 0..9, the default filters=192/khops=2/layers=3
configuration) are published on HuggingFace:
    <PLACEHOLDER: HuggingFace model repo URL>
Download and place them under models/traffic/... matching the layout above
— or train your own with virtual_traffic.py.

    python inference_traffic.py                    # fold 0, default model size
    python inference_traffic.py --fold 3            # fold 3
    python inference_traffic.py --fold 3 --filters 64 --khops 2 --layers 3
    python inference_traffic.py --out results/inference/traffic/fold=3  # also save y/yhat arrays
"""

from data_handling.graph import load_graph_traffic
from data_handling.timeseries import load_data_m2m_traffic, forecast_type
from data_handling.metadata import load_metadata_features
from eval_vis.interence import inference
from virtual_traffic import ids_virtual_fromMetadata

import argparse
import os
import sys
import numpy as np
import torch
import torch_geometric_temporal

FEATURES = [
    'slow', 'fast',
    'VAvg',
    'yearday_sin', 'yearday_cos', 'weektime_sin', 'weektime_cos', 'daytime_sin', 'daytime_cos',
    'praznik_Slovenia', 'praznik_Croatia', 'praznik_Germany', 'praznik_Austria', 'praznik_Switzerland', 'praznik_Italy', 'praznik_Hungary',
    'povp. T', 'kolicina padavin',
    'event_Roadworks'
]
FORECAST_FEATURES = ['slow', 'fast', 'VAvg']
KNOWN_FEATURES = [
    'yearday_sin', 'yearday_cos', 'weektime_sin', 'weektime_cos', 'daytime_sin', 'daytime_cos',
    'praznik_Slovenia', 'praznik_Croatia', 'praznik_Germany', 'praznik_Austria', 'praznik_Switzerland', 'praznik_Italy', 'praznik_Hungary',
    'event_Roadworks'
]
METADATA_FEATURES = [
    'onehot_smer1', 'onehot_smer2',
    'speed_limit',
    'onehot_AC', 'onehot_HC'
]

HORIZON = 6
HISTORY_WINDOW = 23
BATCH_SIZE = 64
SPLIT = 0.666

N_FOLDS = 10  # traffic k-fold data is fixed at 10 folds (0..9)

MODELS_ROOT = 'models/traffic'
HF_MODELS_URL = "<PLACEHOLDER: HuggingFace model repo URL>"


def load_pretrained_model(fold, filters, khops, layers, ctx_features):
    path = f'{MODELS_ROOT}/l={layers}_k={khops}_f={filters}/kfold={fold}.pt'
    if not os.path.isfile(path):
        sys.exit(
            f"error: no checkpoint at {path}\n"
            f"  fold={fold}, filters={filters}, khops={khops}, layers={layers} isn't available locally.\n"
            f"  Download pretrained models from HuggingFace ({HF_MODELS_URL}) and place them\n"
            f"  under {MODELS_ROOT}/... matching that layout, or train your own with virtual_traffic.py."
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
    parser = argparse.ArgumentParser(description='Inference with a pretrained virtual-node GCRNN (traffic).')
    parser.add_argument('--fold', type=int, default=0, help='k-fold split to evaluate on (default: %(default)s)')
    parser.add_argument('--filters', type=int, default=192, help='hidden width of the checkpoint to load (default: %(default)s)')
    parser.add_argument('--khops', type=int, default=2, help='Chebyshev filter order of the checkpoint to load (default: %(default)s)')
    parser.add_argument('--layers', type=int, default=3, help='number of GCRNN layers of the checkpoint to load (default: %(default)s)')
    parser.add_argument('--out', default=None, help='directory to save y/yhat/yv/yhatv .npy arrays (default: print summary only)')
    args = parser.parse_args()

    if not (0 <= args.fold < N_FOLDS):
        sys.exit(f"error: --fold must be in [0, {N_FOLDS - 1}]")

    graph_full = 'data/traffic/graph/stations_expw.nx'
    metadata_full = 'data/traffic/contextual/stations.json'
    metadata_virtual = f"data/traffic/contextual/{args.fold}_virtual.json"
    ids_virtual = ids_virtual_fromMetadata(metadata_virtual, metadata_full)

    edge_weight, edge_index = load_graph_traffic(graph_full, metadata_full)
    features, targets, mean, std = load_data_m2m_traffic(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_only_window, metadata_file=metadata_full)
    ctx = torch.tensor(load_metadata_features(metadata_full, METADATA_FEATURES), dtype=torch.float32)
    dataset = torch_geometric_temporal.StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
    _, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(dataset, train_ratio=SPLIT)

    model = load_pretrained_model(args.fold, args.filters, args.khops, args.layers, ctx.shape[-1])

    y, yhat = inference(model, test_dataset, ctx, BATCH_SIZE, ids_virtual, reconnect_paths=True)

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
        print(f'  {name:20s} MAE (all counters)     = {mae:.4f}')
    for name, mae in zip(FORECAST_FEATURES, summary_v):
        print(f'  {name:20s} MAE (virtual counters) = {mae:.4f}')

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, 'y.npy'), y_np_n)
        np.save(os.path.join(args.out, 'yhat.npy'), yhat_np_n)
        np.save(os.path.join(args.out, 'yv.npy'), y_np_n_virtual)
        np.save(os.path.join(args.out, 'yhatv.npy'), yhat_np_n_virtual)
        print(f'saved y/yhat/yv/yhatv arrays -> {args.out}')


if __name__ == '__main__':
    main()
