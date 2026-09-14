import numpy as np

def build_coord_array(index_dict):
    """
    Converts:
    {idx: {'x': ..., 'y': ...}, ...}
    into:
    indices: np.array of keys
    coords:  np.array shape (N, 2)
    """
    indices = np.array(sorted(index_dict.keys()))
    coords = np.array([
        [index_dict[i]['x'], index_dict[i]['y']]
        for i in indices
    ])
    return indices, coords

def nn(
    yhat,
    metar_reindex,
    metaf_reindex,
    k=1,
    eps=1e-10,
):
    """
    Nearest-Neighbour interpolation baseline.

    For each target node, assigns the value of the k nearest source station(s).
    k=1 is pure NN (Voronoi assignment); k>1 averages the k nearest neighbours
    with equal weights (equivalent to IDW with p→∞ when k=1).

    Parameters
    ----------
    yhat : np.ndarray
        Shape: (samples, old_nodes, timesteps, features)
    metar_reindex : dict
        Existing station coordinates
    metaf_reindex : dict
        Target coordinates
    k : int
        Number of nearest neighbours to average (default 1 = pure NN)
    eps : float
        Distance threshold for exact-match shortcut

    Returns
    -------
    np.ndarray
        Shape: (samples, new_nodes, timesteps, features)
    """
    old_ids, old_xy = build_coord_array(metar_reindex)
    new_ids, new_xy = build_coord_array(metaf_reindex)

    n_old = len(old_ids)
    n_new = len(new_ids)

    assert yhat.shape[1] == n_old

    yhat_interp = np.empty(
        (yhat.shape[0], n_new, yhat.shape[2], yhat.shape[3]),
        dtype=yhat.dtype,
    )

    for ni in range(n_new):
        d = np.sqrt(np.sum((old_xy - new_xy[ni]) ** 2, axis=1))  # (n_old,)

        exact_idx = np.where(d < eps)[0]
        if len(exact_idx) > 0:
            yhat_interp[:, ni] = yhat[:, exact_idx[0]]
            continue

        k_eff = min(k, n_old)
        idx = np.argpartition(d, k_eff)[:k_eff]          # (k,)
        yhat_interp[:, ni] = yhat[:, idx].mean(axis=1)   # (S, T, F)

    return yhat_interp

def tps(
    yhat,
    metar_reindex,
    metaf_reindex,
    reg=1e-6,
    k=None,
    eps=1e-12,
):
    """
    Thin Plate Spline (TPS) interpolation.

    Parameters
    ----------
    yhat : np.ndarray
        Shape:
            (samples, old_nodes, timesteps, features)

    metar_reindex : dict
        Existing station coordinates

    metaf_reindex : dict
        Target coordinates

    reg : float
        Regularization parameter for numerical stability

    k : int or None
        Number of nearest neighbors.
        None = use all stations.

    Returns
    -------
    yhat_interp : np.ndarray
        Shape:
            (samples, new_nodes, timesteps, features)
    """

    # ------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------
    old_ids, old_xy = build_coord_array(metar_reindex)
    new_ids, new_xy = build_coord_array(metaf_reindex)

    n_old = len(old_ids)
    n_new = len(new_ids)

    assert yhat.shape[1] == n_old

    # ------------------------------------------------------------
    # TPS radial basis function
    # phi(r) = r^2 log(r)
    # ------------------------------------------------------------
    def phi(r):
        r = np.maximum(r, eps)
        return (r**2) * np.log(r)

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    yhat_interp = np.empty(
        (yhat.shape[0], n_new, yhat.shape[2], yhat.shape[3]),
        dtype=yhat.dtype
    )

    # ------------------------------------------------------------
    # Interpolate each target node
    # ------------------------------------------------------------
    for ni in range(n_new):

        # --------------------------------------------------------
        # Distances target -> old
        # --------------------------------------------------------
        d = np.sqrt(np.sum((old_xy - new_xy[ni])**2, axis=1))

        # --------------------------------------------------------
        # Exact match
        # --------------------------------------------------------
        exact_idx = np.where(d < eps)[0]

        if len(exact_idx) > 0:
            yhat_interp[:, ni] = yhat[:, exact_idx[0]]
            continue

        # --------------------------------------------------------
        # KNN selection
        # --------------------------------------------------------
        if k is not None and k < n_old:
            idx = np.argpartition(d, k)[:k]
        else:
            idx = np.arange(n_old)

        coords_k = old_xy[idx]

        n_k = len(idx)

        # --------------------------------------------------------
        # Pairwise distances between support points
        # --------------------------------------------------------
        pair_diff = (
            coords_k[:, None, :] -
            coords_k[None, :, :]
        )

        pair_dist = np.sqrt(
            np.sum(pair_diff**2, axis=-1)
        )

        # --------------------------------------------------------
        # TPS kernel matrix
        # --------------------------------------------------------
        K = phi(pair_dist)

        # Regularization
        K += np.eye(n_k) * reg

        # --------------------------------------------------------
        # Polynomial term
        #
        # P = [1, x, y]
        # --------------------------------------------------------
        P = np.concatenate([
            np.ones((n_k, 1)),
            coords_k
        ], axis=1)

        # --------------------------------------------------------
        # Full TPS system
        #
        # [ K  P ]
        # [ Pᵀ 0 ]
        # --------------------------------------------------------
        A = np.zeros((n_k + 3, n_k + 3))

        A[:n_k, :n_k] = K
        A[:n_k, n_k:] = P
        A[n_k:, :n_k] = P.T

        # --------------------------------------------------------
        # Target kernel vector
        # --------------------------------------------------------
        k_target = phi(d[idx])

        p_target = np.array([
            1.0,
            new_xy[ni, 0],
            new_xy[ni, 1]
        ])

        # --------------------------------------------------------
        # Solve TPS weights for ALL values at once
        # --------------------------------------------------------
        #
        # Flatten:
        #   (samples, k, timesteps, features)
        # ->
        #   (k, S*T*F)
        #
        values = np.transpose(
            yhat[:, idx],
            (1, 0, 2, 3)
        ).reshape(n_k, -1)

        rhs = np.concatenate([
            values,
            np.zeros((3, values.shape[1]))
        ], axis=0)

        coeffs = np.linalg.solve(A, rhs)

        w = coeffs[:n_k]
        a = coeffs[n_k:]

        # --------------------------------------------------------
        # Evaluate TPS interpolation
        # --------------------------------------------------------
        interp_flat = (
            k_target @ w +
            p_target @ a
        )

        # --------------------------------------------------------
        # Reshape back
        # --------------------------------------------------------
        yhat_interp[:, ni] = interp_flat.reshape(
            yhat.shape[0],
            yhat.shape[2],
            yhat.shape[3]
        )

    return yhat_interp

def ok(
    yhat,
    metar_reindex,
    metaf_reindex,
    variogram="gaussian",
    range_=100_000.0,
    sill=None,       # if None, inferred from data
    nugget=1e-6,
    k=None,
    eps=1e-10,
):
    """
    Ordinary Kriging interpolation.

    Parameters
    ----------
    yhat : np.ndarray
        Shape: (samples, old_nodes, timesteps, features)

    metar_reindex : dict
        Existing station coordinates

    metaf_reindex : dict
        Target coordinates

    variogram : str
        'gaussian', 'exponential', or 'spherical'

    range_ : float
        Variogram range parameter

    sill : float or None
        Variogram sill. If None, inferred as the empirical variance of yhat
        across (samples, nodes, timesteps), giving one value per feature.

    nugget : float
        Nugget effect / numerical stabilization

    k : int or None
        Number of nearest neighbors

    Returns
    -------
    np.ndarray
        Shape: (samples, new_nodes, timesteps, features)
    """

    # ------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------
    old_ids, old_xy = build_coord_array(metar_reindex)
    new_ids, new_xy = build_coord_array(metaf_reindex)

    n_old = len(old_ids)
    n_new = len(new_ids)

    assert yhat.shape[1] == n_old

    # ------------------------------------------------------------
    # Sill: infer per-feature variance if not provided
    # shape: (features,)  — broadcasts with gamma output later
    # ------------------------------------------------------------
    #if sill is None:
    sill = np.var(yhat, axis=(0, 1, 2))  # (F,)
    # else:
    #     sill = np.full(yhat.shape[-1], sill, dtype=float)  # (F,)

    # ------------------------------------------------------------
    # Variogram models
    # gamma(h) -> (..., F)  for any h shape (...)
    # ------------------------------------------------------------
    def gamma(h):
        h = np.asarray(h)          # (...,)
        h_ = h[..., None]          # (..., 1)  — broadcast over features

        if variogram == "gaussian":
            return nugget + sill * (1.0 - np.exp(-(h_**2) / (range_**2)))

        elif variogram == "exponential":
            return nugget + sill * (1.0 - np.exp(-h_ / range_))

        elif variogram == "spherical":
            hr = h_ / range_
            return np.where(
                h_ <= range_,
                nugget + sill * (1.5 * hr - 0.5 * hr**3),
                nugget + sill,
            )

        else:
            raise ValueError(f"Unknown variogram: {variogram}")

    # ------------------------------------------------------------
    # Old-old distance matrix  (n_old, n_old)
    # ------------------------------------------------------------
    old_diff = old_xy[:, None, :] - old_xy[None, :, :]
    old_dist = np.sqrt(np.sum(old_diff**2, axis=-1))

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    yhat_interp = np.empty(
        (yhat.shape[0], n_new, yhat.shape[2], yhat.shape[3]),
        dtype=yhat.dtype,
    )

    # ------------------------------------------------------------
    # Interpolate each target node separately
    # ------------------------------------------------------------
    for ni in range(n_new):

        # --------------------------------------------------------
        # Distances from target -> old stations  (n_old,)
        # --------------------------------------------------------
        d = np.sqrt(np.sum((old_xy - new_xy[ni]) ** 2, axis=1))

        # --------------------------------------------------------
        # Exact match
        # --------------------------------------------------------
        exact_idx = np.where(d < eps)[0]
        if len(exact_idx) > 0:
            yhat_interp[:, ni] = yhat[:, exact_idx[0]]
            continue

        # --------------------------------------------------------
        # KNN selection
        # --------------------------------------------------------
        if k is not None and k < n_old:
            idx = np.argpartition(d, k)[:k]
        else:
            idx = np.arange(n_old)

        d_k = d[idx]           # (k,)
        coords_k = old_xy[idx] # (k, 2)
        n_k = len(idx)

        # --------------------------------------------------------
        # Kriging system  (k+1, k+1)
        #
        # [ Gamma(h_ij)   1 ]   shape: (k+1, k+1, F)
        # [ 1^T           0 ]
        # --------------------------------------------------------
        pair_diff = coords_k[:, None, :] - coords_k[None, :, :]
        pair_dist = np.sqrt(np.sum(pair_diff**2, axis=-1))  # (k, k)

        Gamma_kk = gamma(pair_dist)   # (k, k, F)
        gamma_0  = gamma(d_k)         # (k, F)

        n_f = yhat.shape[-1]

        # Build (k+1, k+1, F) system, solve F independent systems
        K = np.zeros((n_k + 1, n_k + 1, n_f))
        K[:n_k, :n_k] = Gamma_kk
        K[:n_k, :n_k] += np.eye(n_k)[:, :, None] * nugget  # stabilise
        K[:n_k, -1]   = 1.0
        K[-1, :n_k]   = 1.0

        rhs = np.zeros((n_k + 1, n_f))
        rhs[:n_k] = gamma_0
        rhs[-1]   = 1.0

        # Solve: weights (k+1, F) -> take first k rows -> (k, F)
        weights = np.linalg.solve(
            K.transpose(2, 0, 1),   # (F, k+1, k+1)
            rhs.T,                  # (F, k+1)
        ).T[:n_k]                   # (k, F)

        # --------------------------------------------------------
        # Weighted interpolation
        # yhat[:, idx]:  (S, k, T, F)
        # weights:       (k, F)
        # out:           (S, T, F)
        # --------------------------------------------------------
        yhat_interp[:, ni] = np.einsum(
            'kf,sktf->stf',
            weights,
            yhat[:, idx],
        )

    return yhat_interp

def idw(
    yhat,
    metar_reindex,
    metaf_reindex,
    power=2.0,
    k=None,
    eps=1e-12
):
    """
    IDW interpolation from old stations -> all target stations.

    Parameters
    ----------
    yhat : np.ndarray
        Shape: (samples, old_nodes, timesteps, features)

    metar_reindex : dict
        Existing node coordinates (70 nodes)

    metaf_reindex : dict
        Target node coordinates (78 nodes)

    power : float
        IDW power parameter

    k : int or None
        Number of nearest neighbors to use.
        None = use all old points.

    Returns
    -------
    yhat_interp : np.ndarray
        Shape: (samples, new_nodes, timesteps, features)
    """

    # ------------------------------------------------------------
    # Build coordinate arrays
    # ------------------------------------------------------------
    old_ids, old_xy = build_coord_array(metar_reindex)
    new_ids, new_xy = build_coord_array(metaf_reindex)

    n_old = len(old_ids)
    n_new = len(new_ids)

    assert yhat.shape[1] == n_old, (
        f"yhat has {yhat.shape[1]} nodes but metar_reindex has {n_old}"
    )

    # ------------------------------------------------------------
    # Pairwise distances: (new_nodes, old_nodes)
    # ------------------------------------------------------------
    diff = new_xy[:, None, :] - old_xy[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    # ------------------------------------------------------------
    # Exact matches -> keep original value
    # ------------------------------------------------------------
    exact_match = dist < eps

    # ------------------------------------------------------------
    # Optional KNN restriction
    # ------------------------------------------------------------
    if k is not None and k < n_old:
        knn_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]

        mask = np.ones_like(dist, dtype=bool)
        rows = np.arange(n_new)[:, None]
        mask[rows, knn_idx] = False

        dist = dist.copy()
        dist[mask] = np.inf

    # ------------------------------------------------------------
    # IDW weights
    # ------------------------------------------------------------
    weights = 1.0 / np.power(dist + eps, power)

    # Exact matches get full weight
    for i in range(n_new):
        if np.any(exact_match[i]):
            weights[i] = 0.0
            weights[i, np.argmax(exact_match[i])] = 1.0

    # Normalize
    weights /= weights.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------
    # Interpolate
    # ------------------------------------------------------------
    # yhat shape:
    #   (samples, old_nodes, timesteps, features)
    #
    # weights shape:
    #   (new_nodes, old_nodes)
    #
    # output:
    #   (samples, new_nodes, timesteps, features)
    #
    yhat_interp = np.einsum(
        'no,sotf->sntf',
        weights,
        yhat
    )

    return yhat_interp