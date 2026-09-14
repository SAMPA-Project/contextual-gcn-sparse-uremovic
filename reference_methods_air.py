"""
Classical spatial-interpolation baselines (IDW / kriging / TPS / raster-ConvLSTM)
compared against the base GCRNN, on the air dataset, across k-fold splits
where a subset of stations is held out as "virtual" (unobserved) nodes.

Run with no arguments to run every fold (0..k-1, k=8 by default) through all
three baselines (base_interp, interp_base, interp_raster). Pass --fold to
run only that one fold; pass --k to change the total number of folds (only
meaningful if matching k-fold data was generated with that k).

    python reference_methods_air.py             # all folds
    python reference_methods_air.py --fold 3     # just fold 3
"""

from data_handling.timeseries import load_data_m2m_air
from data_handling.graph import load_graph_traffic
import argparse
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
BATCH_SIZE=64
lr=0.005

SPLIT = 0.666
EPOCHS = 200
REPORT_TRAIN_LOSS_EPOCHS = 1000

def base_interp(k, graph_real, metadata_real, graph_full, metadata_full):
    """
    Baseline: predict on the real (observed-only) graph G-, then spatially
    interpolate (IDW/kriging/TPS) those predictions out to the virtual
    (held-out) nodes of the full graph G. Saves per-method errors.

    Args:
        k:             fold index, used only for naming the results folder
        graph_real:    edgelist path for G- (real stations only)
        metadata_real: station metadata JSON for G-
        graph_full:    edgelist path for G (all stations)
        metadata_full: station metadata JSON for G
    """
    filters = 64
    layers= 2
    khops = 2
    from hyperparams_air import base_model
    y, yhat, _,_ = base_model(graph_real, metadata_real, filters=filters, khops=khops, layers=layers, epochs=EPOCHS, save=False) # predictions on G-
    y_full,_,_,_ = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=1, save=False, skiptrain=True) # labels for G
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

    from data_handling.interp import idw, ok, tps
    yhat_idw = idw(yhat,metar_reindex,metaf_reindex,power=2,k=16)
    yhat_krig = ok(yhat,metar_reindex,metaf_reindex,variogram="spherical",range_=20_000,sill=1.0,nugget=1e-5,k=16)
    yhat_tps = tps(yhat,metar_reindex,metaf_reindex,reg=1e-5,k=16)
    
    real_error = np.mean(np.abs(y - yhat),axis=(0, 1, 2)) # errors on predicted/static
    idw_error = np.mean(np.abs(y_full[:,indices_virtual] - yhat_idw[:,indices_virtual]),axis=(0, 1, 2)) # error on interp idw
    krig_error = np.mean(np.abs(y_full[:,indices_virtual] - yhat_krig[:,indices_virtual]),axis=(0, 1, 2)) # error on interp krig
    tps_error = np.mean(np.abs(y_full[:,indices_virtual] - yhat_tps[:,indices_virtual]),axis=(0, 1, 2)) # error on interp krig

    name = f'base_interp_air/l={layers}_k={khops}_f={filters}/{k}'
    save_dir = f'results/experiments/{name}'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'y_real.npy'), y)
    np.save(os.path.join(save_dir, 'yhat_real.npy'), yhat)
    np.save(f'{save_dir}/yhat_idw_virtual.npy', yhat_idw[:, indices_virtual])
    np.save(f'{save_dir}/yhat_krig_virtual.npy', yhat_krig[:, indices_virtual])
    np.save(f'{save_dir}/yhat_tps_virtual.npy', yhat_tps[:, indices_virtual])
    np.save(f'{save_dir}/y_full_virtual.npy', y_full[:, indices_virtual])

    np.save(f'{save_dir}/{str(real_error)}_real_error.npy', real_error)
    np.save(f'{save_dir}/{str(idw_error)}_idw_error.npy', idw_error)
    np.save(f'{save_dir}/{str(krig_error)}_krig_error.npy', krig_error)
    np.save(f'{save_dir}/{str(tps_error)}_tps_error.npy', tps_error)

def interp_base(k, graph_real, metadata_real, graph_full, metadata_full):
    """
    Baseline: spatially interpolate (IDW/kriging/TPS) the raw inputs/targets
    from G- out to all nodes of G first, then run the base GCRNN on the full
    graph using those interpolated inputs. Saves per-method errors on both
    the real and virtual nodes.

    Args: same as base_interp().
    """
    filters = 64
    layers= 2
    khops = 2
    features, targets, mean, std = load_data_m2m_air(HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES, ft=forecast_type.horizon_window, metadata_file=metadata_real)
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

    from data_handling.interp import idw, ok, tps
    targets_idw = idw(targets,metar_reindex,metaf_reindex,power=2,k=16)
    targets_krig = ok(targets,metar_reindex,metaf_reindex,variogram="spherical",range_=20_000,sill=1.0,nugget=1e-5,k=16)
    targets_tps = tps(targets,metar_reindex,metaf_reindex,reg=1e-5,k=16)
    features_idw = idw(features,metar_reindex,metaf_reindex,power=2,k=16)
    features_krig = ok(features,metar_reindex,metaf_reindex,variogram="spherical",range_=20_000,sill=1.0,nugget=1e-5,k=16)
    features_tps = tps(features,metar_reindex,metaf_reindex,reg=1e-5,k=16)
    from hyperparams_air import base_model
    y, _, _, _ = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=1, save=False, skiptrain=True) # true labels G
    _, yhat_idw, _, _ = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=EPOCHS, save=False, interp_values=(features_idw, targets_idw))
    _, yhat_krig, _, _ = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=EPOCHS, save=False, interp_values=(features_krig, targets_krig))
    _, yhat_tps, _, _ = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers, epochs=EPOCHS, save=False, interp_values=(features_tps, targets_tps))

    idw_real_error = np.mean(np.abs(y[:,indices_real] - yhat_idw[:,indices_real]),axis=(0, 1, 2))
    krig_real_error = np.mean(np.abs(y[:,indices_real] - yhat_krig[:,indices_real]),axis=(0, 1, 2))
    tps_real_error = np.mean(np.abs(y[:,indices_real] - yhat_tps[:,indices_real]),axis=(0, 1, 2))
    idw_error = np.mean(np.abs(y[:,indices_virtual] - yhat_idw[:,indices_virtual]),axis=(0, 1, 2))
    krig_error = np.mean(np.abs(y[:,indices_virtual] - yhat_krig[:,indices_virtual]),axis=(0, 1, 2))
    tps_error = np.mean(np.abs(y[:,indices_virtual] - yhat_tps[:,indices_virtual]),axis=(0, 1, 2))

    name = f'interp_base_air/l={layers}_k={khops}_f={filters}/{k}'
    save_dir = f'results/experiments/{name}'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'y_real.npy'), y[:,indices_real])
    np.save(f'{save_dir}/y_full_virtual.npy', y[:, indices_virtual])
    np.save(os.path.join(save_dir, 'yhat_idw_real.npy'), yhat_idw[:,indices_real])
    np.save(os.path.join(save_dir, 'yhat_krig_real.npy'), yhat_krig[:,indices_real])
    np.save(os.path.join(save_dir, 'yhat_tps_real.npy'), yhat_tps[:,indices_real])
    np.save(f'{save_dir}/yhat_idw_virtual.npy', yhat_idw[:, indices_virtual])
    np.save(f'{save_dir}/yhat_krig_virtual.npy', yhat_krig[:, indices_virtual])
    np.save(f'{save_dir}/yhat_tps_virtual.npy', yhat_tps[:, indices_virtual])
    
    np.save(f'{save_dir}/{str(idw_real_error)}_idw_real_error.npy', idw_real_error)
    np.save(f'{save_dir}/{str(krig_real_error)}_krig_real_error.npy', krig_real_error)
    np.save(f'{save_dir}/{str(tps_real_error)}_tps_real_error.npy', tps_real_error)
    np.save(f'{save_dir}/{str(idw_error)}_idw_error.npy', idw_error)
    np.save(f'{save_dir}/{str(krig_error)}_krig_error.npy', krig_error)
    np.save(f'{save_dir}/{str(tps_error)}_tps_error.npy', tps_error)

def interp_raster(i, graph_real, metadata_real, graph_full, metadata_full):
    """
    Baseline: spatially interpolate the sparse per-station data onto a
    raster grid, train a ConvLSTM context-encoder-decoder on that grid, then
    read predictions back off at station pixel locations. Saves per-method
    (idw/ok/tps) errors on both the real and virtual nodes.

    Args: same as base_interp(), with `i` used both for naming and as the
    fold index (interchangeable with `k` there).
    """
    pixel_size = 480 * 8
    from data_handling.grid import full_grid
    (X0, X1, Y0, Y1), rasters = full_grid(pixel_size, metadata_real)
 
    # sparse features/targets for training (sparsified for memory)
    features, targets, mean, std = load_data_m2m_air(
        HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES,
        ft=forecast_type.hrzowd_sparse, metadata_file=metadata_real,
    )
    # full features/targets for inference (aligned to y_full_gt)
    # _, targets_full, _, _ = load_data_m2m(
    #     HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES,
    #     ft=forecast_type.horizon_only_window, metadata_file=metadata_full,
    # )
    features_nonsparse, _, _, _ = load_data_m2m_air(
        HISTORY_WINDOW, HORIZON, FEATURES, FORECAST_FEATURES,
        ft=forecast_type.horizon_only_window, metadata_file=metadata_real,
    )
 
    # ── full-graph ground truth ───────────────────────────────────────────────
    from hyperparams_air import base_model
    y_full_gt, _, _, _ = base_model(
        graph_full, metadata_full, filters=8, khops=1, layers=1,
        epochs=1, save=False, skiptrain=True,
    )  # (S_test, N_full, H, F_out)
 
    import json
    with open(metadata_real, 'r') as fp:
        metadatar = json.load(fp)
    metar_reindex = {
        metadatar[n]['idx']: {'x': metadatar[n]['x'], 'y': metadatar[n]['y']}
        for n in metadatar
    }
    with open(metadata_full, 'r') as fp:
        metadataf = json.load(fp)
    metaf_reindex = {
        metadataf[n]['idx']: {'x': metadataf[n]['x'], 'y': metadataf[n]['y']}
        for n in metadataf
    }
 
    # ── grid ──────────────────────────────────────────────────────────────────
    x_grid = np.arange(X0, X1, pixel_size)
    y_grid = np.arange(Y1, Y0, -pixel_size)
    grid_w, grid_h = len(x_grid), len(y_grid)
 
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_reindex = {
        j: {'x': float(xx.flat[j]), 'y': float(yy.flat[j])}
        for j in range(grid_h * grid_w)
    }
 
    def to_grid(arr):
        # (S, grid_h*grid_w, T, F) -> (S, T, F, grid_h, grid_w)
        arr = arr.reshape(arr.shape[0], grid_h, grid_w, arr.shape[2], arr.shape[3])
        return arr.transpose(0, 3, 4, 1, 2)
 
    # ── station -> pixel mapping ──────────────────────────────────────────────
    def station_to_grid_idx(reindex, x_grid, y_grid):
        xs   = np.array([v['x'] for v in reindex.values()])
        ys   = np.array([v['y'] for v in reindex.values()])
        cols = np.argmin(np.abs(xs[:, None] - x_grid[None, :]), axis=1)
        rows = np.argmin(np.abs(ys[:, None] - y_grid[None, :]), axis=1)
        return rows, cols
 
    rows_full, cols_full = station_to_grid_idx(metaf_reindex, x_grid, y_grid)
 
    old_ids         = set(metar_reindex.keys())
    all_ids         = sorted(metaf_reindex.keys())
    indices_virtual = [j for j, sid in enumerate(all_ids) if sid not in old_ids]
    indices_real    = [j for j, sid in enumerate(all_ids) if sid     in old_ids]
 
    # ── sparse training set (from G- / sparse loader) ────────────────────────
    from data_handling.interp import idw, ok, tps
 
    def to_grid(arr):
        arr = arr.reshape(arr.shape[0], grid_h, grid_w, arr.shape[2], arr.shape[3])
        return arr.transpose(0, 3, 4, 1, 2)
 
    S_sparse = features.shape[0]
    split_sparse = int(S_sparse * 0.666)
    tr_features = features[:split_sparse]
    tr_targets  = targets[:split_sparse]
    te_features = features[split_sparse:]
    te_targets = targets[split_sparse:]
 
    def interp_sparse(arr, method):
        if method == 'idw': return to_grid(idw(arr, metar_reindex, grid_reindex, power=1.5, k=16))
        if method == 'ok':  return to_grid(ok( arr, metar_reindex, grid_reindex, variogram="spherical", range_=20_000, nugget=1e-5, k=16))
        if method == 'tps': return to_grid(tps(arr, metar_reindex, grid_reindex, reg=1e-5, k=16))
 
    tr_x = {m: interp_sparse(tr_features, m) for m in ('idw', 'ok', 'tps')}
    tr_y = {m: interp_sparse(tr_targets,  m) for m in ('idw', 'ok', 'tps')}
    te_x = {m: interp_sparse(te_features, m) for m in ('idw', 'ok', 'tps')}
    te_y = {m: interp_sparse(te_targets,  m) for m in ('idw', 'ok', 'tps')}

    # ── full test set split — aligned to y_full_gt ───────────────────────────
    # targets_full[test_start:test_end] aligns to y_full_gt at atol=1e-5
    S_full      = features_nonsparse.shape[0]
    test_start  = int(S_full * 0.666)
    test_end    = S_full - 20           # -20 matches base_model's internal trim
    S_test      = test_end - test_start
 
    assert S_test == y_full_gt.shape[0], \
        f"test set size mismatch: {S_test} vs y_full_gt {y_full_gt.shape[0]}"
 
    te_features_full = features_nonsparse[test_start:test_end]   # (S_test, N_full, T, F)
 
    # ── context ───────────────────────────────────────────────────────────────
    ctx = rasters.transpose(2, 0, 1).astype(np.float32)     # (C_ctx, H, W)
 
    # ── model factory ─────────────────────────────────────────────────────────
    import torch
    from models_impl.convlstm import ConvLSTM_ctxCNN_encoderdecoder
    from training.training import train_raster
 
    def make_model():
        return ConvLSTM_ctxCNN_encoderdecoder(
            filters           = 16,
            layers            = 2,
            node_features     = len(FEATURES),
            forecast_features = len(FORECAST_FEATURES),
            known_features    = len(KNOWN_FEATURES),
            ctx_features      = ctx.shape[0],
            history           = HISTORY_WINDOW,
            horizon           = HORIZON,
        )
 
    loss_fn    = torch.nn.MSELoss()
    BATCH_SIZE = 32
    EPOCHS_    = 100
 
    # ── save dir ──────────────────────────────────────────────────────────────
    save_dir = f'results/experiments/interp_raster_air/fold={i}'
    os.makedirs(save_dir, exist_ok=True)
    #np.save(f'{save_dir}/y_full_gt.npy', y_full_gt)
 
    # ── train, chunked inference, evaluate ───────────────────────────────────
    for name in ('idw', 'ok', 'tps'):
        model = make_model()
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        exp   = f'interp_raster_air/{name}/fold={i}'
 
        # train on sparse grid interpolations
        train_raster(exp, model, tr_x[name], te_x[name], tr_y[name], te_y[name],
                     ctx, loss_fn, opt, EPOCHS_, BATCH_SIZE)
 
        # ── chunked inference on full test set ────────────────────────────────
        # interpolate chunk by chunk to avoid holding all S_test grid frames in RAM
        yhat_chunks = []
 
        device = torch.device(0)
        model.eval()
        model.to(device)
        ctx_b  = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0) \
                     .expand(BATCH_SIZE, -1, -1, -1).to(device)

        with torch.no_grad():
            for c_start in range(0, S_test, BATCH_SIZE):
                chunk = te_features_full[c_start:c_start + BATCH_SIZE]  # (B, N, T, F)
 
                # interpolate this chunk to grid on the fly
                x_chunk = to_grid(
                    idw(chunk, metar_reindex, grid_reindex, power=1.5, k=16)
                    if name == 'idw' else
                    ok( chunk, metar_reindex, grid_reindex, variogram="spherical", range_=20_000, nugget=1e-5, k=16)
                    if name == 'ok' else
                    tps(chunk, metar_reindex, grid_reindex, reg=1e-5, k=16)
                )  # (B, T, C, H, W)
 
                x_t   = torch.tensor(x_chunk, dtype=torch.float32).to(device)
                y_hat = model(x_t, ctx_b)   # (B, H, C_out, H, W)
 
                # map grid predictions -> station nodes immediately (saves memory)
                # y_hat_nodes: (B, H, C_out, N_full) -> (B, N_full, H, C_out)
                y_hat_nodes = y_hat[:, :, :, rows_full, cols_full].permute(0, 3, 1, 2)
                yhat_chunks.append(y_hat_nodes.cpu().numpy())
 
                print(f'{name}  chunk {c_start//BATCH_SIZE+1}/{S_test//BATCH_SIZE}')
 
        # (S_eval, N_full, H, C_out)
        yhat_at_stations = np.concatenate(yhat_chunks, axis=0)
        S_eval           = min(yhat_at_stations.shape[0], y_full_gt.shape[0])
        yhat_at_stations = yhat_at_stations[:S_eval] * std[:len(FORECAST_FEATURES)] + mean[:len(FORECAST_FEATURES)]
        gt               = y_full_gt[:S_eval]
 
        assert yhat_at_stations.shape == gt.shape, \
            f"shape mismatch: yhat {yhat_at_stations.shape} vs gt {gt.shape}"
 
        error_real    = np.mean(np.abs(gt[:, indices_real]    - yhat_at_stations[:, indices_real]),    axis=(0, 1, 2))
        error_virtual = np.mean(np.abs(gt[:, indices_virtual] - yhat_at_stations[:, indices_virtual]), axis=(0, 1, 2))
 
        np.save(f'{save_dir}/y_full_virtual.npy', gt[:, indices_virtual])
        np.save(f'{save_dir}/y_real.npy', gt[:, indices_real])
        np.save(f'{save_dir}/yhat_{name}_virtual.npy', yhat_at_stations[:, indices_virtual])
        np.save(f'{save_dir}/yhat_{name}_real.npy', yhat_at_stations[:, indices_real])
        np.save(f'{save_dir}/{str(error_real)}_{name}_real_error.npy',    error_real)
        np.save(f'{save_dir}/{str(error_virtual)}_{name}_virtual_error.npy', error_virtual)
 
        print(f'{name}  real_error={error_real}  virtual_error={error_virtual}')

def main():
    parser = argparse.ArgumentParser(description='Interpolation baselines vs. base GCRNN (air), across k-fold splits.')
    parser.add_argument('--k', type=int, default=8,
                         help='total number of folds the k-fold data was generated with (default: %(default)s)')
    parser.add_argument('--fold', type=int, default=None,
                         help='run only this fold index; default runs every fold 0..k-1')
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else range(args.k)
    for i in folds:
        graph_real = f'data/air/graph/{i}_real_inv.nx'
        metadata_real = f"data/air/contextual/{i}_real.json"
        metadata_virtual = f"data/air/contextual/{i}_virtual.json"
        graph_full = "data/air/graph/stations_inv.nx"
        metadata_full = 'data/air/contextual/stations.json'
        base_interp(i, graph_real, metadata_real, graph_full, metadata_full)
        interp_base(i, graph_real, metadata_real, graph_full, metadata_full)
        interp_raster(i, graph_real, metadata_real, graph_full, metadata_full)

if __name__ == '__main__':
    main()