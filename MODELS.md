# Pretrained models

Pretrained checkpoints for the virtual-node GCRNN (`GCRNNVirtual`, the
proposed method — see [`virtual_meteo.py`](virtual_meteo.py)/
[`virtual_traffic.py`](virtual_traffic.py)/[`virtual_air.py`](virtual_air.py))
are published on HuggingFace:

**`<PLACEHOLDER: HuggingFace model repo URL>`**

Download and extract to `models/` in the repository root, matching this
layout exactly:

```
models/<dataset>/l=<layers>_k=<khops>_f=<filters>/kfold=<fold>.pt
```

Only one (layers, khops, filters) configuration is published per dataset —
the one the paper's results use:

| dataset | layers | khops | filters | folds |
|---|---|---|---|---|
| meteo   | 3 | 2 | 128 | 0–8 |
| traffic | 3 | 2 | 192 | 0–9 |
| air     | 3 | 2 | 64  | 0–7 |

Each `.pt` is a plain `torch.save(model.state_dict())` for `GCRNNVirtual`
constructed with that dataset's fixed architecture (feature counts, history/
horizon window — see the corresponding `virtual_*.py`) and the (layers,
khops, filters) in its path. `kfold=<n>` matches the k-fold split it was
trained on — i.e. the checkpoint was trained on that fold's real-only graph
G- with a subset of nodes masked as virtual, so its virtual-node metrics
reported by `inference_*.py` reflect genuinely held-out stations/counters.

## Running inference

`inference_meteo.py`, `inference_traffic.py`, and `inference_air.py` each
load one checkpoint and evaluate it on that fold's test split, reporting
mean absolute error overall and on the virtual (held-out) nodes only:

```bash
python inference_meteo.py                                   # fold 0, default model size
python inference_traffic.py --fold 3                         # fold 3, default model size
python inference_air.py --fold 3 --filters 64 --khops 2 --layers 3
python inference_meteo.py --out results/inference/meteo/fold=0  # also save y/yhat/yv/yhatv .npy
```

If the requested (`--fold`, `--filters`, `--khops`, `--layers`) combination
isn't available under `models/<dataset>/...`, the script tells you exactly
which checkpoint path it looked for and points you back at the HuggingFace
repo above, instead of failing with a raw file-not-found error.

## Collecting checkpoints from an old training run (maintainer note)

If you have the original `phd_scripts/results/experiments/phd/...` training
outputs and need to (re-)build the `models/` tree to publish, there's a
`collect_hf_models.py` script for that — it's a private one-off tool (reads
from that old, non-public path layout) and isn't tracked in this repo. Ask
whoever ran the training for it, or reconstruct it from the mapping in the
table above: each `models/<dataset>/l=..._k=..._f=.../kfold=<n>.pt` comes
from `phd_scripts/results/experiments/phd/<vn_name>/vr=<vr>/l=..._k=..._f=..._lr=<lr>/<n>/model.pt`,
where `<vn_name>` is `vn_residual_1000_horizonfix` (meteo), `vn_dars`
(traffic), or `vn_arso_big` (air), and `<vr>`/`<lr>` are that dataset's
`virtual_ratio`/learning rate (see the table above and each `virtual_*.py`'s
module docstring for the exact defaults).
