import numpy as np
from collections import deque


# =============================================================================
# GRAPH UTILITIES
# =============================================================================

def bfs_hop_distance(edge_index, edge_weight, n_nodes, source, max_hops, sigma=5487.1):
    hop_dist  = np.full(n_nodes, np.inf)
    road_dist = np.zeros(n_nodes)       # 0.0 = unreachable (log(0) = -inf -> dist=inf)
    hop_dist[source]  = 0
    road_dist[source] = 1.0             # multiplicative identity: exp(-0/sigma) = 1

    adj = {i: [] for i in range(n_nodes)}
    for k in range(edge_index.shape[1]):
        u, v = edge_index[0, k], edge_index[1, k]
        w    = float(edge_weight[k])
        adj[u].append((v, w))
        adj[v].append((u, w))

    queue = deque([source])
    while queue:
        u = queue.popleft()
        if hop_dist[u] >= max_hops:
            continue
        for v, w in adj[u]:
            new_hops = hop_dist[u] + 1
            new_road = road_dist[u] * w      # product of weights along path
            if new_hops < hop_dist[v]:
                hop_dist[v]  = new_hops
                road_dist[v] = new_road
                queue.append(v)
            elif new_hops == hop_dist[v] and new_road > road_dist[v]:
                # higher product = stronger connection = better path
                road_dist[v] = new_road

    # convert back to distance: w = exp(-d/sigma) => d = -sigma * log(w)
    # unreachable nodes have road_dist=0.0 -> log(0)=-inf -> dist=inf (correct)
    with np.errstate(divide='ignore'):
        dist = -sigma * np.log(road_dist)

    return hop_dist.astype(np.float64), dist.astype(np.float64)


def get_neighbourhood(edge_index, edge_weight, n_nodes, source, max_hops):
    hop_dist, road_dist = bfs_hop_distance(
        edge_index, edge_weight, n_nodes, source, max_hops
    )
    mask = (hop_dist > 0) & (hop_dist <= max_hops)
    idx  = np.where(mask)[0].astype(np.int64)   # <-- force int dtype
    return idx, road_dist[idx]


# =============================================================================
# GRAPH-TOPOLOGY IDW
# =============================================================================

def idw_graph(
    yhat,
    source_nodes,
    target_nodes,
    edge_weight,
    edge_index,
    n_nodes=220,
    power=2.0,
    max_hops=3,
    eps=1e-12,
):
    """
    IDW interpolation using graph road-distance.

    Parameters
    ----------
    yhat         : (S, N_source, H, F)
    target_nodes : list/array of target node indices (into the full graph)
    source_nodes : list/array of source node indices (into the full graph),
                   must correspond to yhat axis=1
    edge_index   : (2, E) int array
    edge_weight  : (E,) float array — road lengths
    n_nodes      : total number of nodes in the full graph
    power        : IDW power
    max_hops     : neighbourhood radius in hops

    Returns
    -------
    (S, N_target, H, F)
    """
    if isinstance(target_nodes, dict):
        target_nodes = np.array(sorted(target_nodes.keys()))
    if isinstance(source_nodes, dict):
        source_nodes = np.array(sorted(source_nodes.keys()))

    S, N_src, H, F = yhat.shape
    N_tgt = len(target_nodes)
    out   = np.empty((S, N_tgt, H, F), dtype=yhat.dtype)

    # map full-graph index -> yhat axis-1 position
    full_to_src = {int(n): i for i, n in enumerate(source_nodes)}

    for ti, t in enumerate(target_nodes):
        # exact match: target is also a source
        if t in full_to_src:
            out[:, ti] = yhat[:, full_to_src[t]]
            continue

        nbr_idx, road_dists = get_neighbourhood(
            edge_index, edge_weight, n_nodes, int(t), max_hops
        )
        # keep only neighbours that are source nodes
        src_mask = np.array([n in full_to_src for n in nbr_idx], dtype=bool)
        nbr_src  = nbr_idx[src_mask] if len(nbr_idx) > 0 else np.array([], dtype=np.int64)
        nbr_dists = road_dists[src_mask]

        if len(nbr_src) == 0:
            # no source within max_hops — fall back to globally nearest source
            # compute road distance via BFS from t to all sources
            _, rd = bfs_hop_distance(edge_index, edge_weight, n_nodes, int(t), n_nodes)
            src_rd   = rd[source_nodes]
            nbr_src  = source_nodes
            nbr_dists = src_rd

        w = 1.0 / (nbr_dists + eps) ** power
        w /= w.sum()

        src_pos = np.array([full_to_src[int(n)] for n in nbr_src])
        # yhat[:, src_pos]: (S, k, H, F); weights: (k,)
        out[:, ti] = np.einsum('k,skhf->shf', w, yhat[:, src_pos])

    return out


# =============================================================================
# GRAPH-TOPOLOGY TPS
# =============================================================================

def tps_graph(
    yhat,
    source_nodes,
    target_nodes,
    edge_weight,
    edge_index,
    n_nodes=220,
    reg=1e-6,
    max_hops=3,
    eps=1e-12,
):
    """
    Thin Plate Spline interpolation using graph road-distance as the metric.

    The TPS kernel is defined as phi(r) = r^2 * log(r) where r is the
    road-distance between support points.  The polynomial term is dropped
    (only the RBF part is used) since there are no meaningful 2-D coordinates
    to define a linear trend on a graph.

    Parameters
    ----------
    yhat         : (S, N_source, H, F)
    target_nodes : array of target node indices
    source_nodes : array of source node indices (correspond to yhat axis=1)
    edge_index   : (2, E) int
    edge_weight  : (E,) float — road lengths
    n_nodes      : total nodes in full graph
    reg          : regularisation for numerical stability
    max_hops     : neighbourhood radius in hops

    Returns
    -------
    (S, N_target, H, F)
    """
    if isinstance(target_nodes, dict):
        target_nodes = np.array(sorted(target_nodes.keys()))
    if isinstance(source_nodes, dict):
        source_nodes = np.array(sorted(source_nodes.keys()))

    S, N_src, H, F = yhat.shape
    N_tgt = len(target_nodes)
    out   = np.empty((S, N_tgt, H, F), dtype=yhat.dtype)

    full_to_src = {int(n): i for i, n in enumerate(source_nodes)}

    def phi(r):
        r = np.maximum(r, eps)
        return (r ** 2) * np.log(r)

    # precompute road distances between all source node pairs
    # (expensive once, reused for every target)
    src_road = np.zeros((N_src, N_src))
    for i, s in enumerate(source_nodes):
        _, rd = bfs_hop_distance(
            edge_index, edge_weight, n_nodes, int(s), n_nodes
        )
        src_road[i] = rd[source_nodes]

    K_src = phi(src_road) + np.eye(N_src) * reg   # (N_src, N_src)

    for ti, t in enumerate(target_nodes):
        if t in full_to_src:
            out[:, ti] = yhat[:, full_to_src[t]]
            continue

        nbr_idx, _ = get_neighbourhood(
            edge_index, edge_weight, n_nodes, int(t), max_hops
        )
        src_mask = np.array([n in full_to_src for n in nbr_idx], dtype=bool)
        nbr_src  = nbr_idx[src_mask] if len(nbr_idx) > 0 else np.array([], dtype=np.int64)

        if len(nbr_src) < 3:
            nbr_src = source_nodes   # fall back to all sources

        src_pos = np.array([full_to_src[int(n)] for n in nbr_src])
        n_k     = len(src_pos)

        # road distances: target -> each support node
        _, rd_t = bfs_hop_distance(
            edge_index, edge_weight, n_nodes, int(t), n_nodes
        )
        k_target = phi(rd_t[nbr_src])           # (n_k,)

        # sub-matrix for the selected support nodes
        K = K_src[np.ix_(src_pos, src_pos)]     # (n_k, n_k)

        # solve for all (S, H, F) values at once
        vals = np.transpose(yhat[:, src_pos], (1, 0, 2, 3)).reshape(n_k, -1)
        w    = np.linalg.solve(K, vals)         # (n_k, S*H*F)

        interp = k_target @ w                   # (S*H*F,)
        out[:, ti] = interp.reshape(S, H, F)

    return out

def nn_graph(
    yhat,
    source_nodes,
    target_nodes,
    edge_weight,
    edge_index,
    n_nodes=220,
    max_hops=1,
    eps=1e-10,
):
    """
    Nearest-Neighbour interpolation using graph road-distance.

    For each target node, finds the source node with the shortest road-distance
    path within max_hops. Falls back to globally nearest source if none found.

    Parameters
    ----------
    yhat         : (S, N_source, H, F)
    target_nodes : array of target node indices (full graph)
    source_nodes : array of source node indices (full graph, correspond to yhat axis=1)
    edge_index   : (2, E) int
    edge_weight  : (E,) float — road lengths
    n_nodes      : total nodes in full graph
    max_hops     : neighbourhood radius in hops

    Returns
    -------
    (S, N_target, H, F)
    """
    if isinstance(target_nodes, dict):
        target_nodes = np.array(sorted(target_nodes.keys()))
    if isinstance(source_nodes, dict):
        source_nodes = np.array(sorted(source_nodes.keys()))

    S, N_src, H, F = yhat.shape
    N_tgt = len(target_nodes)
    out   = np.empty((S, N_tgt, H, F), dtype=yhat.dtype)

    full_to_src = {int(n): i for i, n in enumerate(source_nodes)}

    for ti, t in enumerate(target_nodes):
        if t in full_to_src:
            out[:, ti] = yhat[:, full_to_src[t]]
            continue

        nbr_idx, road_dists = get_neighbourhood(
            edge_index, edge_weight, n_nodes, int(t), max_hops
        )
        src_mask = np.array([n in full_to_src for n in nbr_idx], dtype=bool)
        nbr_src  = nbr_idx[src_mask] if len(nbr_idx) > 0 else np.array([], dtype=np.int64)
        nbr_dists = road_dists[src_mask]

        if len(nbr_src) == 0:
            # fall back to globally nearest source
            _, rd = bfs_hop_distance(edge_index, edge_weight, n_nodes, int(t), n_nodes)
            nbr_src   = source_nodes
            nbr_dists = rd[source_nodes]

        nearest = full_to_src[int(nbr_src[np.argmin(nbr_dists)])]
        out[:, ti] = yhat[:, nearest]

    return out

# =============================================================================
# GRAPH-TOPOLOGY ORDINARY KRIGING
# =============================================================================

def ok_graph(
    yhat,
    source_nodes,
    target_nodes,
    edge_weight,
    edge_index,
    n_nodes=220,
    variogram="spherical",
    range_=5.0,         # in hops (or road-distance units if you prefer)
    nugget=1e-6,
    max_hops=3,
    eps=1e-10,
):
    """
    Ordinary Kriging using graph road-distance.

    Parameters
    ----------
    yhat         : (S, N_source, H, F)
    target_nodes : array of target node indices
    source_nodes : array of source node indices
    edge_index   : (2, E) int
    edge_weight  : (E,) float
    n_nodes      : total nodes in full graph
    variogram    : 'gaussian', 'exponential', or 'spherical'
    range_       : variogram range (same units as edge_weight)
    nugget       : nugget / numerical stabilisation
    max_hops     : neighbourhood radius

    Returns
    -------
    (S, N_target, H, F)
    """
    if isinstance(target_nodes, dict):
        target_nodes = np.array(sorted(target_nodes.keys()))
    if isinstance(source_nodes, dict):
        source_nodes = np.array(sorted(source_nodes.keys()))

    S, N_src, H, F = yhat.shape
    N_tgt = len(target_nodes)
    out   = np.empty((S, N_tgt, H, F), dtype=yhat.dtype)

    full_to_src = {int(n): i for i, n in enumerate(source_nodes)}

    sill = np.var(yhat, axis=(0, 1, 2))   # (F,)

    def gamma(h):
        h  = np.asarray(h)
        h_ = h[..., None]          # (..., 1) broadcast over features
        if variogram == "gaussian":
            return nugget + sill * (1.0 - np.exp(-(h_ ** 2) / (range_ ** 2)))
        elif variogram == "exponential":
            return nugget + sill * (1.0 - np.exp(-h_ / range_))
        elif variogram == "spherical":
            hr = h_ / range_
            return np.where(
                h_ <= range_,
                nugget + sill * (1.5 * hr - 0.5 * hr ** 3),
                nugget + sill,
            )
        else:
            raise ValueError(f"Unknown variogram: {variogram}")

    # precompute road distances between all source pairs
    src_road = np.zeros((N_src, N_src))
    for i, s in enumerate(source_nodes):
        _, rd = bfs_hop_distance(
            edge_index, edge_weight, n_nodes, int(s), n_nodes
        )
        src_road[i] = rd[source_nodes]

    Gamma_src = gamma(src_road)   # (N_src, N_src, F)

    for ti, t in enumerate(target_nodes):
        if t in full_to_src:
            out[:, ti] = yhat[:, full_to_src[t]]
            continue

        nbr_idx, _ = get_neighbourhood(
            edge_index, edge_weight, n_nodes, int(t), max_hops
        )
        src_mask = np.array([n in full_to_src for n in nbr_idx], dtype=bool)
        nbr_src  = nbr_idx[src_mask] if len(nbr_idx) > 0 else np.array([], dtype=np.int64)

        if len(nbr_src) < 2:
            nbr_src = source_nodes

        src_pos = np.array([full_to_src[int(n)] for n in nbr_src])
        n_k     = len(src_pos)

        _, rd_t = bfs_hop_distance(
            edge_index, edge_weight, n_nodes, int(t), n_nodes
        )
        gamma_0 = gamma(rd_t[nbr_src])           # (n_k, F)

        # kriging system (n_k+1, n_k+1, F)
        G_kk = Gamma_src[np.ix_(src_pos, src_pos)]   # (n_k, n_k, F)
        K = np.zeros((n_k + 1, n_k + 1, F))
        K[:n_k, :n_k] = G_kk
        K[:n_k, :n_k] += np.eye(n_k)[:, :, None] * nugget
        K[:n_k, -1]    = 1.0
        K[-1, :n_k]    = 1.0

        rhs = np.zeros((n_k + 1, F))
        rhs[:n_k] = gamma_0
        rhs[-1]   = 1.0

        # solve F independent systems -> weights (n_k, F)
        weights = np.linalg.solve(
            K.transpose(2, 0, 1),    # (F, n_k+1, n_k+1)
            rhs.T,                   # (F, n_k+1)
        ).T[:n_k]                    # (n_k, F)

        out[:, ti] = np.einsum(
            'kf,skhf->shf',
            weights,
            yhat[:, src_pos],
        )

    return out