"""
Run this on the machine where `data/` was already built by an older version
of collect_zenodo_data.py (the "phd_scripts"-era layout: `stations_curated_
sorted*.json`, `graph/<name>/edgelist.nx`, `metadata/k_fold/...`, etc.) —
migrates it in place to the current unified layout:

    data/<dataset>/graph/...        edgelists (.nx), flat files
    data/<dataset>/contextual/...   metadata jsons, raster npys, context csvs
    data/<dataset>/timeseries/...   per-node timeseries csvs (unchanged names)

and file naming: `stations.json`/`stations*.nx` for the full set, `<fold>_
real`/`<fold>_virtual` for k-fold subsets, graph files named after their
matching metadata plus a weighting-scheme suffix where more than one graph
variant exists for the same node set.

Also deletes `data/paper/` and `data/mag/` if present — unrelated to this
project's data.

This only ever *moves* files already on disk (no network access, unlike
collect_zenodo_data.py which pulls from the original `tools/` tree) so it's
safe to re-run; anything already migrated or already absent is skipped.

Usage:
    python repack_data.py [--root data] [--dry-run]

    --root      the data/ directory to migrate in place (default: ./data)
    --dry-run   print what would move/delete without touching anything
"""

import argparse
import shutil
import sys
from pathlib import Path


def move(src: Path, dst: Path, dry_run: bool, moved: list, missing: list):
    if not src.is_file():
        missing.append(src)
        return
    if dst.exists():
        moved.append((src, dst, "skipped (destination already exists)"))
        return
    moved.append((src, dst, None))
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def move_dir_contents(src_dir: Path, dst_dir: Path, dry_run: bool, moved: list, missing: list):
    if not src_dir.is_dir():
        missing.append(src_dir)
        return
    for f in sorted(src_dir.iterdir()):
        if f.is_file():
            move(f, dst_dir / f.name, dry_run, moved, missing)


def remove_dir(path: Path, dry_run: bool, removed: list):
    if path.exists():
        removed.append(path)
        if not dry_run:
            shutil.rmtree(path)


def build_moves(root: Path):
    """Returns (single_file_moves, dir_moves) for all three datasets."""

    single = []
    dirs = []

    # ── meteo ────────────────────────────────────────────────────────────────
    m = root / "meteo"
    single += [
        (m / "stations_curated_sorted.json", m / "contextual/stations.json"),
        (m / "dtm/dtm.npy", m / "contextual/dtm.npy"),
        (m / "sentinel/sentinel2_data.npy", m / "contextual/sentinel2_data.npy"),
        (m / "graph/elev_distance_3500_distance_30000_corr_0975_fullHigherBenchmarks/edgelist.nx",
         m / "graph/stations.nx"),
        (m / "graph/elev_distance_3500_distance_30000_corr_0975/edgelist.nx",
         m / "graph/stations_hp.nx"),
    ]
    for i in range(9):
        single += [
            (m / f"stations_curated_sorted_REAL_K=8_i={i}.json", m / f"contextual/{i}_real.json"),
            (m / f"stations_curated_sorted_VIRTUAL_k=8_i={i}.json", m / f"contextual/{i}_virtual.json"),
            (m / f"graph/elev_distance_3500_distance_30000_corr_0975_REAL_K=8_i={i}/edgelist.nx",
             m / f"graph/{i}_real.nx"),
        ]
    dirs.append((m / "timeseries", m / "timeseries"))  # already correctly placed; no-op

    # ── traffic ──────────────────────────────────────────────────────────────
    t = root / "traffic"
    single += [
        (t / "counters_consolidated.json", t / "contextual/stations.json"),
        (t / "graph/consolidated/edgelist.nx", t / "graph/stations.nx"),
        (t / "graph/consolidated/edgelist_expw.nx", t / "graph/stations_expw.nx"),
    ]
    for i in range(10):
        single += [
            (t / f"metadata/k_fold/{i}.json", t / f"contextual/{i}_real.json"),
            (t / f"metadata/k_fold/{i}_v.json", t / f"contextual/{i}_virtual.json"),
            (t / f"graph/k_fold/{i}/edgelist.nx", t / f"graph/{i}_real.nx"),
            (t / f"graph/k_fold/{i}/edgelist_expw.nx", t / f"graph/{i}_real_expw.nx"),
        ]
    dirs.append((t / "preproc/consolidated", t / "timeseries"))

    # ── air ──────────────────────────────────────────────────────────────────
    a = root / "air"
    single += [
        (a / "stations_curated.json", a / "contextual/stations.json"),
        (a / "graph/full/edgelist_inv.nx", a / "graph/stations_inv.nx"),
        (a / "metadata/contextual_features_normalized.csv",
         a / "contextual/contextual_features_normalized.csv"),
    ]
    for i in range(8):
        single += [
            (a / f"metadata/k_fold/{i}.json", a / f"contextual/{i}_real.json"),
            (a / f"metadata/k_fold/{i}_v.json", a / f"contextual/{i}_virtual.json"),
            (a / f"graph/k_fold/{i}/edgelist_inv.nx", a / f"graph/{i}_real_inv.nx"),
        ]
    dirs.append((a / "enriched", a / "timeseries"))

    return single, dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data", help="the data/ directory to migrate in place (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen without touching anything")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    single_moves, dir_moves = build_moves(root)

    moved, missing = [], []
    for src, dst in single_moves:
        move(src, dst, args.dry_run, moved, missing)
    for src_dir, dst_dir in dir_moves:
        if src_dir == dst_dir:
            continue
        move_dir_contents(src_dir, dst_dir, args.dry_run, moved, missing)

    prefix = "[dry-run] would move" if args.dry_run else "moved"
    real_moves = [(s, d) for s, d, note in moved if note is None]
    skipped = [(s, d, note) for s, d, note in moved if note is not None]

    print(f"{prefix} {len(real_moves)} files")
    for s, d, note in skipped:
        print(f"  SKIPPED [{note}] {s} -> {d}")
    if missing:
        print(f"\n{len(missing)} source files not found (already migrated, or never fetched):")
        for p in missing:
            print(f"  {p}")

    # ── drop unrelated directories ────────────────────────────────────────────
    removed = []
    for junk in ("paper", "mag"):
        remove_dir(root / junk, args.dry_run, removed)
    if removed:
        verb = "[dry-run] would remove" if args.dry_run else "removed"
        print(f"\n{verb} {len(removed)} unrelated director" + ("y" if len(removed) == 1 else "ies") + ":")
        for p in removed:
            print(f"  {p}")

    if not args.dry_run:
        # clean up now-empty leftover directories from the old layout
        for dataset in ("meteo", "traffic", "air"):
            for leftover in ("graph", "metadata", "dtm", "sentinel", "preproc", "enriched"):
                d = root / dataset / leftover
                if d.is_dir():
                    # deepest first, so a parent is only removed after its now-empty children are
                    for sub in sorted((p for p in d.rglob("*") if p.is_dir()),
                                       key=lambda p: len(p.parts), reverse=True):
                        if sub.is_dir() and not any(sub.iterdir()):
                            sub.rmdir()
                    if d.is_dir() and not any(d.iterdir()) and leftover != "graph":
                        d.rmdir()

    print("\ndone." if not args.dry_run else "\ndry-run done, nothing was changed.")


if __name__ == "__main__":
    main()
