import torch
import numpy as np


def virtualize_graph(
    x, y, ei, ew, station_context,
    ids_virtual, nnodes, batch_size, device, layers,
    rnnfilters=64,
    reconnect=False,
):
    """
    Masks virtual nodes out of a diagonally-batched graph.

    Args:
        x               : (batch*nnodes, T, F)
        y               : (batch*nnodes, H or T+H, F_out)
        ei              : (2, E)   global batched edge index
        ew              : (E,)     edge weights
        station_context : (batch*nnodes, maps, H_grid, W_grid)
        ids_virtual     : sorted list of node indices in [0, nnodes) to mask
        nnodes          : number of nodes in one graph
        batch_size      : number of graphs in the diagonal batch
        device          : torch device
        layers          : number of RNN layers
        rnnfilters      : hidden state width

    Returns:
        h_                           : zero hidden state list, (batch*nnodes, rnnfilters) per layer
        x_real                       : (n_real*batch, T, F)
        y                            : (batch*nnodes, H or T+H, F_out) reordered real-first
        y_real                       : (n_real*batch, H, F_out)
        ei_real                      : (2, E_real)
        ew_real                      : (E_real,)
        station_context_real         : (n_real*batch, maps, H_grid, W_grid)
        mask_real_diagonal_batched   : bool list, length batch*nnodes
        mask_virtual_diagonal_batched: bool list, length batch*nnodes
    """
    H = 12

    # ── node masks ────────────────────────────────────────────────────────────
    ids_virtual_set = set(ids_virtual)
    mask_virtual = [i in ids_virtual_set for i in range(nnodes)]
    mask_real    = [not v for v in mask_virtual]
    ids_real     = [i for i in range(nnodes) if not mask_virtual[i]]

    mask_virtual_diagonal_batched = mask_virtual * batch_size
    mask_real_diagonal_batched    = mask_real    * batch_size

    assert x.shape[0] == station_context.shape[0] == len(mask_real_diagonal_batched), \
        "Node count mismatch between x, station_context and mask"

    # ── real node inputs ──────────────────────────────────────────────────────
    x_real               = x[mask_real_diagonal_batched].clone()
    station_context_real = station_context[mask_real_diagonal_batched].clone()

    # ── y: split, optionally zero virtual history, then reorder ──────────────
    y_real    = y[mask_real_diagonal_batched]
    y_virtual = y[mask_virtual_diagonal_batched]

    if y_virtual.shape[1] > H:
        y_virtual = y_virtual.clone()
        y_virtual[:, :-H, :] = 0

    y = torch.cat([y_real, y_virtual], dim=0)
    y_real = y_real[:, -H:, :]

    # ── edge re-indexing ──────────────────────────────────────────────────────
    n_real = len(ids_real)
    mapto  = np.arange(nnodes)
    for v in ids_virtual:
        mapto[v + 1:] -= 1

    mapto_full = np.concatenate([mapto + n_real * i for i in range(batch_size)])

    # ── edge re-indexing ──────────────────────────────────────────────────────
    ei_np  = ei.numpy()
    ew_np  = ew.numpy()

    if reconnect:
        ei_np, ew_np,_ = reconnect_virtual_nodes(
            ei_np, ew_np, ids_virtual_set, nnodes
        )
    src, dst = ei_np[0], ei_np[1]

    ei_mask_real = np.array([
        mask_real_diagonal_batched[a] and mask_real_diagonal_batched[b]
        for a, b in zip(src, dst)
    ])

    src_r = mapto_full[src[ei_mask_real]]
    dst_r = mapto_full[dst[ei_mask_real]]

    ei_real = torch.tensor(np.stack([src_r, dst_r]), dtype=torch.int64)
    ew_real = torch.tensor(ew_np[ei_mask_real]).clone()

    # ── pre-allocated hidden state container ─────────────────────────────────
    h_ = [torch.zeros(x.shape[0], rnnfilters, device=device) for _ in range(layers)]

    x_real               = x_real.to(device)
    ei_real              = ei_real.to(device)
    ew_real              = ew_real.to(device)
    station_context_real = station_context_real.to(device)

    return (
        h_,
        x_real,
        y,
        y_real,
        ei_real,
        ew_real,
        station_context_real,
        mask_real_diagonal_batched,
        mask_virtual_diagonal_batched,
    )


def reconnect_virtual_nodes(ei_np, ew_np, ids_virtual, n_nodes):
    """
    Remove virtual nodes one at a time.
    For each virtual node v:
      1. For every pair (u, v, w) where u->v and v->w both exist
         and u,w are not v itself: add edge u->w with weight ew(u->v)+ew(v->w)
         (only if u->w doesn't already exist)
      2. Remove v and all edges touching v from the graph
    Handles chains of consecutive virtual nodes correctly.

    Returns
    -------
    ei_np, ew_np  : cleaned edge arrays (no virtual nodes)
    bridging_set  : set of (u, w) pairs that were added as bridges
    """

    # work with lists for dynamic insertion/deletion
    edges = {}  # (u, v) -> weight
    for k in range(ei_np.shape[1]):
        u, v = int(ei_np[0, k]), int(ei_np[1, k])
        edges[(u, v)] = float(ew_np[k])

    bridging_set = set()

    for v in ids_virtual:
        # find all predecessors (u->v) and successors (v->w)
        predecessors = {u: w for (u, vv), w in edges.items() if vv == v}
        successors   = {w: ww for (vv, w), ww in edges.items() if vv == v}

        # bridge every predecessor to every successor
        for u, w_uv in predecessors.items():
            for w, w_vw in successors.items():
                if u == w:
                    continue
                if (u, w) not in edges:
                    new_weight = w_uv * w_vw
                    edges[(u, w)] = new_weight
                    bridging_set.add((u, w))

        # remove v and all edges touching v
        to_delete = [key for key in edges if key[0] == v or key[1] == v]
        for key in to_delete:
            del edges[key]

    # rebuild numpy arrays
    if edges:
        src_arr = np.array([k[0] for k in edges.keys()], dtype=ei_np.dtype)
        dst_arr = np.array([k[1] for k in edges.keys()], dtype=ei_np.dtype)
        ew_arr  = np.array(list(edges.values()),          dtype=ew_np.dtype)
        ei_out  = np.stack([src_arr, dst_arr])
    else:
        ei_out = np.empty((2, 0), dtype=ei_np.dtype)
        ew_arr = np.empty((0,),   dtype=ew_np.dtype)

    return ei_out, ew_arr, bridging_set