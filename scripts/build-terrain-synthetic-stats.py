#!/usr/bin/env python3
"""Build synthetic_map_stats.json — the quantile tables the conditioning map needs.

terrain-diffusion conditions its coarse stage on a synthetic climate map: five
Perlin FBm fields (elevation, temperature, temperature seasonality, precipitation,
precipitation seasonality) each quantile-matched onto the marginal distribution of
the REAL Earth. That matching is what makes the fields look like a planet instead
of like noise, and it needs global raster data to derive.

Upstream computes those tables lazily on first run and caches them to
`data/global/synthetic_map_stats.json`. That file is NOT in the checkout, and
producing it needs rasterio plus a 10-arc-minute WorldClim download — neither of
which we want as a runtime dependency of a C++ terrain generator. So we do it
once, here, offline, and ship the ~640 resulting numbers next to the weights.

This script deliberately *imports and calls upstream's own `_compute_map_stats`*
rather than reimplementing it. Reimplementing would mean our tables were only ever
checked against our own reading of upstream's statistics code; running upstream's
code means the tables are upstream's by construction, and the C++ port then only
has to match the far smaller sampling/finalize surface (which terrain_synthetic_parity.sh
does cover).

Inputs:
  ../terrain-diffusion/data/global/etopo_10m.tif   (ships in the checkout)
  wc2.1_10m_bio_{1,4,12,15}.tif                    (downloaded here, ~130 MB zip)

Usage: scripts/build-terrain-synthetic-stats.py [--out-dir DIR] [--td-root DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

WC_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
WC_FILES = [
    "wc2.1_10m_bio_1.tif",
    "wc2.1_10m_bio_4.tif",
    "wc2.1_10m_bio_12.tif",
    "wc2.1_10m_bio_15.tif",
]

# These come from the CHECKPOINT's config.json, never from a default in the
# source. WorldPipeline's constructor signature says [1.5, 3, 3, 3, 3] / 0.5, but
# from_pretrained overrides it with the shipped config, and the 30 m checkpoint
# ships [1.0]*5 / 0.5 — so the signature defaults are what the pipeline runs with
# only if you construct it by hand. The tables depend on both (frequency sets the
# noise quantiles, drop_water_pct reshapes the elevation histogram by discarding
# half the ocean), and the 90 m checkpoint is free to differ, so read them rather
# than baking in either set.
FALLBACK_FREQUENCY_MULT = [1.0, 1.0, 1.0, 1.0, 1.0]
FALLBACK_DROP_WATER_PCT = 0.5


def pipeline_constants(config_path: Path):
    """(frequency_mult, drop_water_pct) from a shipped WorldPipeline config.json."""
    if not config_path.exists():
        print(f"==> no {config_path}, falling back to {FALLBACK_FREQUENCY_MULT} / "
              f"{FALLBACK_DROP_WATER_PCT}")
        return FALLBACK_FREQUENCY_MULT, FALLBACK_DROP_WATER_PCT
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    # A shipped config may legitimately omit a key, in which case the pipeline
    # really does fall back to its constructor default of None -> [1.5, 3, 3, 3, 3].
    freq = cfg.get("frequency_mult") or [1.5, 3, 3, 3, 3]
    drop = cfg.get("drop_water_pct")
    if drop is None:
        drop = 0.5
    return list(freq), float(drop)


def ensure_worldclim(global_dir: Path) -> None:
    missing = [f for f in WC_FILES if not (global_dir / f).exists()]
    if not missing:
        print("==> WorldClim rasters present")
        return
    global_dir.mkdir(parents=True, exist_ok=True)
    zip_path = global_dir / "wc2.1_10m_bio.zip"
    print(f"==> downloading {WC_URL}")
    urllib.request.urlretrieve(WC_URL, zip_path)
    print("==> extracting")
    with zipfile.ZipFile(zip_path) as zf:
        # Extract only the four bands we need; the archive holds all 19.
        for name in WC_FILES:
            with zf.open(name) as src, open(global_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink()
    print("==> done")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--td-root", default="../terrain-diffusion")
    ap.add_argument("--out-dir", default=None,
                    help="default: weights/terrain-diffusion-30m-bro")
    ap.add_argument("--config", default=None,
                    help="shipped pipeline config.json to read frequency_mult / "
                         "drop_water_pct from (default: weights/terrain-diffusion-30m)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    td = Path(args.td_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir \
        else repo / "weights" / "terrain-diffusion-30m-bro"
    out_path = out_dir / "synthetic_map_stats.json"
    config_path = Path(args.config).resolve() if args.config \
        else repo / "weights" / "terrain-diffusion-30m" / "config.json"

    if out_path.exists() and not args.force:
        print(f"==> {out_path} exists (use --force to rebuild)")
        return 0

    if not (td / "terrain_diffusion").is_dir():
        print(f"error: no terrain-diffusion checkout at {td}", file=sys.stderr)
        return 2

    global_dir = td / "data" / "global"
    if not (global_dir / "etopo_10m.tif").exists():
        print(f"error: {global_dir / 'etopo_10m.tif'} missing — it ships in the "
              f"terrain-diffusion checkout; is the clone complete (git-lfs)?",
              file=sys.stderr)
        return 2
    ensure_worldclim(global_dir)

    frequency_mult, drop_water_pct = pipeline_constants(config_path)

    # Upstream reads its rasters through paths relative to the CWD, and writes its
    # cache to data/global/, so run from the checkout root and copy the result out.
    sys.path.insert(0, str(td))
    os.chdir(td)
    from terrain_diffusion.inference import synthetic_map as sm

    print(f"==> computing stats (frequency_mult={frequency_mult}, "
          f"drop_water_pct={drop_water_pct}) from {config_path}")
    stats = sm._compute_map_stats(frequency_mult, drop_water_pct)
    sm._save_stats_cache(stats)

    src = td / sm.STATS_CACHE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"==> wrote {out_path}")
    print(f"    n_quantiles      {payload['n_quantiles']}")
    print(f"    noise tables     {len(payload['noise_quantile_tables'])}")
    print(f"    data tables      {len(payload['data_quantile_tables'])}")
    print(f"    a_temp_std       {payload['a_temp_std']:.8f}")
    print(f"    b_temp_std       {payload['b_temp_std']:.8f}")
    print(f"    temp_std_p1/p99  {payload['temp_std_p1']:.6f} {payload['temp_std_p99']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
