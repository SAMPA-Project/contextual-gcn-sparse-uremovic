"""
Run this on the machine that still has the original `tools/` data tree
(the larger ml-research project) — NOT in this repo's checkout.

Collects exactly the files the scripts in this repo actually read (see
DATA.md for the full description of each), copies them into a `data/`
directory laid out the way this repo expects, and zips the result.

Usage:
    python collect_zenodo_data.py [--source-root tools] [--dest-root data] [--no-zip]

    --source-root   directory containing meteo_slo/, dars_traffic/, arso_air/
                     (default: ./tools, i.e. the original relative paths)
    --dest-root     where to build the data/ tree (default: ./data)
    --no-zip        skip zipping dest-root at the end

Every copy is best-effort: a missing file is reported at the end rather
than aborting the run, so you can see exactly what's missing and decide
whether to source it another way.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def copy_file(src: Path, dst: Path, missing: list, optional: bool = False, note: str = ""):
    if not src.is_file():
        if not optional:
            missing.append((src, note))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_file_with_fallback(candidates, dst: Path, missing: list, note: str = ""):
    """Try each candidate source path in order; copy the first that exists."""
    for src in candidates:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
    missing.append((candidates[0], f"{note} (also tried: {', '.join(str(c) for c in candidates[1:])})"))
    return False


def copy_dir_files(src_dir: Path, dst_dir: Path, pattern: str, missing: list, note: str = ""):
    """Non-recursive: copies only files directly inside src_dir matching pattern."""
    if not src_dir.is_dir():
        missing.append((src_dir, note))
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)
            n += 1
    if n == 0:
        missing.append((src_dir / pattern, note + " (matched 0 files)"))
    return n


def build_manifest(src_root: Path, dst_root: Path):
    """Returns (static_files, fallback_files, dir_globs, fold_files) — see main() for shapes."""

    # ── meteo ────────────────────────────────────────────────────────────────
    meteo_src = src_root / "meteo_slo"
    meteo_dst = dst_root / "meteo"

    static_files = [
        (meteo_src / "data/full_curated/SamodejnePostaje_curated_sorted.json",
         meteo_dst / "stations_curated_sorted.json"),
        (meteo_src / "dtm/data/dtm.npy",
         meteo_dst / "dtm/dtm.npy"),
        (meteo_src / "sentinel/data/sentinel2_data.npy",
         meteo_dst / "sentinel/sentinel2_data.npy"),
    ]

    # named graph variants, each a directory containing edgelist.nx
    meteo_graph_names = [
        "elev_distance_3500_distance_30000_corr_0975",
        "elev_distance_3500_distance_30000_corr_0975_fullHigherBenchmarks",
    ] + [
        f"elev_distance_3500_distance_30000_corr_0975_REAL_K=8_i={i}" for i in range(9)
    ]
    for name in meteo_graph_names:
        static_files.append((
            meteo_src / f"data/full_curated/graph/{name}/edgelist.nx",
            meteo_dst / f"graph/{name}/edgelist.nx",
        ))

    # meteo k-fold metadata (k=8, 9 folds: i=0..8)
    fold_files = []
    for i in range(9):
        fold_files.append((
            meteo_src / f"data/full_curated/SamodejnePostaje_curated_sorted_REAL_K=8_i={i}.json",
            meteo_dst / f"stations_curated_sorted_REAL_K=8_i={i}.json",
        ))
        fold_files.append((
            meteo_src / f"data/full_curated/SamodejnePostaje_curated_sorted_VIRTUAL_k=8_i={i}.json",
            meteo_dst / f"stations_curated_sorted_VIRTUAL_k=8_i={i}.json",
        ))

    dir_globs = [
        (meteo_src / "data/full_curated/timeseries", meteo_dst / "timeseries", "*"),
    ]

    # ── traffic ──────────────────────────────────────────────────────────────
    traffic_src = src_root / "dars_traffic"
    traffic_dst = dst_root / "traffic"

    static_files += [
        (traffic_src / "counters_consolidated.json",
         traffic_dst / "counters_consolidated.json"),
        (traffic_src / "data/graph_consolidated/edgelist.nx",
         traffic_dst / "graph/consolidated/edgelist.nx"),
    ]
    # generated (not raw) — copy if present, but don't complain if missing;
    # data_handling/generate_expw_graphs.py can (re)build these
    optional_files = [
        (traffic_src / "data/graph_consolidated/edgelist_expw.nx",
         traffic_dst / "graph/consolidated/edgelist_expw.nx"),
    ]

    for i in range(10):
        fold_files.append((
            traffic_src / f"metadata/k_fold/{i}.json",
            traffic_dst / f"metadata/k_fold/{i}.json",
        ))
        fold_files.append((
            traffic_src / f"metadata/k_fold/{i}_v.json",
            traffic_dst / f"metadata/k_fold/{i}_v.json",
        ))
        fold_files.append((
            traffic_src / f"data/k_fold/{i}/edgelist.nx",
            traffic_dst / f"graph/k_fold/{i}/edgelist.nx",
        ))
        optional_files.append((
            traffic_src / f"data/k_fold/{i}/edgelist_expw.nx",
            traffic_dst / f"graph/k_fold/{i}/edgelist_expw.nx",
        ))

    dir_globs.append(
        (traffic_src / "data/preproc/consolidated", traffic_dst / "preproc/consolidated", "*.csv")
    )

    # ── air ──────────────────────────────────────────────────────────────────
    air_src = src_root / "arso_air"
    air_dst = dst_root / "air"

    # two different paths were used for this file across the original scripts;
    # try both (see DATA.md notes) rather than guessing which one is real
    fallback_files = [
        ([air_src / "arso_air_postaje_curated.json",
          air_src / "data/metadata/arso_air_postaje_curated.json"],
         air_dst / "stations_curated.json"),
    ]

    static_files += [
        (air_src / "data/graph/edgelist_inv.nx",
         air_dst / "graph/full/edgelist_inv.nx"),
        (air_src / "metadata/contextual_features_normalized.csv",
         air_dst / "metadata/contextual_features_normalized.csv"),
    ]

    for i in range(8):
        fold_files.append((
            air_src / f"metadata/k_fold_paper/{i}.json",
            air_dst / f"metadata/k_fold/{i}.json",
        ))
        fold_files.append((
            air_src / f"metadata/k_fold_paper/{i}_v.json",
            air_dst / f"metadata/k_fold/{i}_v.json",
        ))
        fold_files.append((
            air_src / f"data/k_fold_paper/{i}/edgelist_inv.nx",
            air_dst / f"graph/k_fold/{i}/edgelist_inv.nx",
        ))

    dir_globs.append(
        (air_src / "data/enriched", air_dst / "enriched", "*.csv")
    )

    return static_files, optional_files, fallback_files, fold_files, dir_globs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", default="tools", help="directory containing meteo_slo/, dars_traffic/, arso_air/ (default: %(default)s)")
    parser.add_argument("--dest-root", default="data", help="where to build the data/ tree (default: %(default)s)")
    parser.add_argument("--no-zip", action="store_true", help="don't zip dest-root at the end")
    args = parser.parse_args()

    src_root = Path(args.source_root)
    dst_root = Path(args.dest_root)

    if not src_root.is_dir():
        print(f"error: source root {src_root} does not exist", file=sys.stderr)
        sys.exit(1)

    static_files, optional_files, fallback_files, fold_files, dir_globs = build_manifest(src_root, dst_root)

    missing = []
    copied = 0

    for src, dst in static_files + fold_files:
        if copy_file(src, dst, missing, note="required"):
            copied += 1

    for src, dst in optional_files:
        if copy_file(src, dst, missing, optional=True):
            copied += 1

    for candidates, dst in fallback_files:
        if copy_file_with_fallback(candidates, dst, missing, note="required (tried multiple known paths)"):
            copied += 1

    for src_dir, dst_dir, pattern in dir_globs:
        copied += copy_dir_files(src_dir, dst_dir, pattern, missing, note="required directory")

    print(f"\ncopied {copied} files into {dst_root}/")

    if missing:
        print(f"\n{len(missing)} missing (see below) — fix paths/--source-root and re-run, or source these separately:")
        for path, note in missing:
            print(f"  MISSING [{note}] {path}")
    else:
        print("nothing missing.")

    if not args.no_zip:
        zip_path = dst_root.with_suffix(".zip")
        print(f"\nzipping {dst_root}/ -> {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(dst_root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(dst_root.parent))
        print(f"done -> {zip_path}")

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
