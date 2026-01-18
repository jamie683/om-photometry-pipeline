from __future__ import annotations

import argparse
import csv
import json
import platform
import getpass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u


# -----------------------------------------------------------------------------
# DEFAULT CONFIG (used unless CLI overrides)
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "science_dir": r"C:\Users\Jamie McAteer\PythonSaves\Fits\TOI_4546_b\data",
    "bias_dir":    r"C:\Users\Jamie McAteer\PythonSaves\Fits\TOI_4546_b\bias",
    "outdir":      r"C:\Users\Jamie McAteer\PythonSaves\Fits\TOI_4546_b\reduction",

    # Optional
    "camera_settings": r"C:\Users\Jamie McAteer\PythonSaves\Fits\TOI_4546_b\TOI_00001.CameraSettings.txt",

    # Target / observatory (only needed for BJD_TDB)
    "target_ra":  None,          # "10:00:00.0"
    "target_dec": None,          # "+10:00:00.0"
    "obs_lat":    None,          # degrees
    "obs_lon":    None,          # degrees (East +)
    "obs_height": 0.0,           # meters
}



# -----------------------------------------------------------------------------
# 0) Mini logbook (auto-appended every run)
# -----------------------------------------------------------------------------
def log_update(log_path: Path, title: str, details: str = "", meta: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    user = getpass.getuser()
    host = platform.node()

    lines = []
    lines.append("=" * 78)
    lines.append(f"{ts} | {title}")
    lines.append(f"User/Host: {user}@{host}")
    if details.strip():
        lines.append("")
        lines.append(details.rstrip())
    if meta:
        lines.append("")
        lines.append("Meta:")
        lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# 1) FITS read + header normalization
# -----------------------------------------------------------------------------
def read_asi2600mc_fits(fits_path: Path):
    with fits.open(fits_path, memmap=False) as hdul:
        h = hdul[0].header
        img = np.asarray(hdul[0].data)

    meta = {
        "path": str(fits_path),
        "date_obs": h.get("DATE-OBS"),
        "date_avg": h.get("DATE-AVG"),
        "date_end": h.get("DATE-END"),
        "jd_utc": h.get("JD_UTC"),
        "exptime": float(h.get("EXPTIME", np.nan)),
        "imagetype": (h.get("IMAGETYP") or h.get("FRAMETYP") or "").strip(),
        "object": (h.get("OBJECT") or "").strip(),
        "instrume": (h.get("INSTRUME") or "").strip(),
        "bayerpat": (h.get("BAYERPAT") or "").strip(),
        "roworder": (h.get("ROWORDER") or "").strip(),
        "adcbits": h.get("ADCBITS"),
        "egain_e_per_adu": float(h.get("EGAIN", np.nan)),  # SharpCap label: "Electrons per ADU"
        "rdnoise_e": float(h.get("RDNOISE", np.nan)),
        "offset": h.get("OFFSET"),
        "blklevel": h.get("BLKLEVEL"),
        "biasadu": h.get("BIASADU"),
        "xbin": h.get("XBINNING", 1),
        "ybin": h.get("YBINNING", 1),
        "naxis1": h.get("NAXIS1"),
        "naxis2": h.get("NAXIS2"),
    }
    return img, h, meta


# -----------------------------------------------------------------------------
# 2) Bayer -> photometry image (Green average for RGGB)
# -----------------------------------------------------------------------------
def extract_green_from_rggb(raw: np.ndarray) -> np.ndarray:
    # RGGB tile:
    # R G
    # G B
    # Two greens: [0,1] and [1,0]
    g1 = raw[0::2, 1::2]
    g2 = raw[1::2, 0::2]
    return 0.5 * (g1.astype(np.float32) + g2.astype(np.float32))


# -----------------------------------------------------------------------------
# 3) Time conversion: mid-exposure UTC -> BJD_TDB (optional but recommended)
# -----------------------------------------------------------------------------
def compute_bjd_tdb(meta: dict, target: SkyCoord, location: EarthLocation) -> float:
    # Prefer SharpCap mid-exposure JD_UTC if present; else DATE-AVG
    jd_utc = meta.get("jd_utc", None)
    if jd_utc is not None and np.isfinite(jd_utc):
        t = Time(float(jd_utc), format="jd", scale="utc", location=location)
    elif meta.get("date_avg"):
        t = Time(meta["date_avg"], format="isot", scale="utc", location=location)
    else:
        return float("nan")

    ltt = t.light_travel_time(target, kind="barycentric")
    return float((t.tdb + ltt).jd)


# -----------------------------------------------------------------------------
# 4) Master bias builder (median stack)
# -----------------------------------------------------------------------------
def build_master_bias(bias_dir: Path, outdir: Path, log_path: Path) -> np.ndarray:
    bias_files = sorted(list(bias_dir.rglob("*.fit*")))
    if not bias_files:
        raise RuntimeError(f"No bias FITS found in: {bias_dir}")

    stack = []
    for f in bias_files:
        try:
            img, _, _ = read_asi2600mc_fits(f)
            stack.append(img.astype(np.float32))
        except Exception as e:
            log_update(log_path, "Skipped bias frame", details=str(e), meta={"file": str(f)})

    if len(stack) < 3:
        raise RuntimeError(f"Too few readable bias frames: {len(stack)} (need >= 3)")

    master = np.nanmedian(np.stack(stack, axis=0), axis=0).astype(np.float32)

    # Save master bias as FITS
    out_fits = outdir / "master_bias.fits"
    hdu = fits.PrimaryHDU(master)
    hdu.header["HISTORY"] = "Median master bias (OM SharpCap ASI2600MC)"
    fits.HDUList([hdu]).writeto(out_fits, overwrite=True)

    log_update(
        log_path,
        "Built master bias",
        details=f"- bias_dir: {bias_dir}\n- used: {len(stack)} frames\n- saved: {out_fits.name}",
        meta={"shape": list(master.shape)}
    )

    return master


# -----------------------------------------------------------------------------
# 5) Main reduction: bias-only + green extraction + optional BJD_TDB
# -----------------------------------------------------------------------------
def reduce_om_dataset(
    science_dir: Path,
    bias_dir: Path,
    outdir: Path,
    camera_settings_txt: Path | None,
    target_ra: str | None,
    target_dec: str | None,
    obs_lat: float | None,
    obs_lon: float | None,
    obs_height_m: float,
):
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "reduction_logbook.txt"

    log_update(
        log_path,
        "Reduction run started",
        details=(
            f"- science_dir: {science_dir}\n"
            f"- bias_dir: {bias_dir}\n"
            f"- outdir: {outdir}\n"
            f"- darks: NONE (not applied)\n"
            f"- flats: NONE (not applied)\n"
            f"- calibration: bias-only\n"
        ),
    )

    # Optional: copy camera settings into output and log it
    if camera_settings_txt and camera_settings_txt.exists():
        copied = outdir / camera_settings_txt.name
        copied.write_text(camera_settings_txt.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        log_update(log_path, "Camera settings captured", details=f"- copied: {copied.name}")

    # Build master bias
    master_bias = None
    if bias_dir is not None and Path(bias_dir).exists():
        try:
            master_bias = build_master_bias(Path(bias_dir), outdir, log_path)
        except Exception as e:
            log_update(log_path, "Master bias not built", details=str(e))
    else:
        log_update(log_path, "Bias directory missing/unused", details=f"- bias_dir: {bias_dir}")

    # Optional BJD inputs
    have_bjd = all([target_ra, target_dec, obs_lat is not None, obs_lon is not None])
    if have_bjd:
        target = SkyCoord(target_ra, target_dec, unit=(u.hourangle, u.deg))
        location = EarthLocation(lat=obs_lat * u.deg, lon=obs_lon * u.deg, height=obs_height_m * u.m)
        log_update(log_path, "BJD_TDB enabled", details="- using astropy barycentric correction")
    else:
        target = None
        location = None
        log_update(log_path, "BJD_TDB disabled", details="- missing target coords and/or observatory location")

    # Gather science frames
    sci_files = sorted(list(science_dir.rglob("*.fit*")))
    if not sci_files:
        raise RuntimeError(f"No science FITS found in: {science_dir}")

    # Preallocate cube list
    green_frames = []
    rows = []

    for i, f in enumerate(sci_files):
        try:
            raw, hdr, meta = read_asi2600mc_fits(f)

            # Only process Light frames if IMAGETYP exists
            if meta.get("imagetype") and meta["imagetype"].lower() not in ("light", "science"):
                continue

            # Bias-only calibration (float32)
            cal = raw.astype(np.float32)
            if master_bias is not None:
                cal = cal - master_bias
            else:
                # no bias available: proceed uncalibrated
                pass
            

            # Bayer handling: must be RGGB for this extractor
            bpat = (meta.get("bayerpat") or "").upper()
            if bpat not in ("RGGB", ""):
                # If SharpCap flips pattern sometimes, you can add handlers later.
                log_update(log_path, "Unexpected Bayer pattern", meta={"file": str(f), "bayerpat": bpat})

            green = extract_green_from_rggb(cal)
            green_frames.append(green)

            # Timing
            bjd_tdb = float("nan")
            if have_bjd:
                bjd_tdb = compute_bjd_tdb(meta, target, location)

            rows.append({
                "idx": i,
                "filename": f.name,
                "path": str(f),
                "exptime_s": meta.get("exptime", float("nan")),
                "jd_utc": meta.get("jd_utc", float("nan")),
                "date_avg": meta.get("date_avg", ""),
                "bjd_tdb": bjd_tdb,
                "bayerpat": meta.get("bayerpat", ""),
                "roworder": meta.get("roworder", ""),
                "gain_setting": hdr.get("GAIN", ""),
                "offset": hdr.get("OFFSET", ""),
                "blklevel": hdr.get("BLKLEVEL", ""),
                "ccd_temp_C": hdr.get("CCD-TEMP", ""),
            })

        except Exception as e:
            log_update(log_path, "Skipped science frame", details=str(e), meta={"file": str(f)})
            continue

    if len(green_frames) < 3:
        raise RuntimeError(f"Too few processed science frames: {len(green_frames)}")

    # Write cube FITS (N, ny, nx)
    cube = np.stack(green_frames, axis=0).astype(np.float32)
    cube_path = outdir / "OM_green_cube.fits"
    hdu = fits.PrimaryHDU(cube)
    hdu.header["HISTORY"] = "Bias-only calibrated; green-plane extracted from RGGB Bayer mosaic"
    fits.HDUList([hdu]).writeto(cube_path, overwrite=True)

    # Manifest CSV
    manifest_path = outdir / "manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log_update(
        log_path,
        "Reduction run finished",
        details=(
            f"- processed_frames: {len(green_frames)}\n"
            f"- cube: {cube_path.name} (shape={cube.shape})\n"
            f"- manifest: {manifest_path.name}\n"
        ),
    )

    print(f"\nDone.\n  Cube:     {cube_path}\n  Manifest: {manifest_path}\n  Logbook:  {log_path}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="OM bias-only reduction for SharpCap ASI2600MC Pro FITS.",
        add_help=True
    )

    # All args optional — CLI overrides config
    p.add_argument("--science-dir", type=str, default=None)
    p.add_argument("--bias-dir", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--camera-settings", type=str, default=None)

    p.add_argument("--target-ra", type=str, default=None)
    p.add_argument("--target-dec", type=str, default=None)
    p.add_argument("--obs-lat", type=float, default=None)
    p.add_argument("--obs-lon", type=float, default=None)
    p.add_argument("--obs-height", type=float, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Merge CLI overrides onto DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG.copy()
    for k in cfg:
        cli_val = getattr(args, k.replace("-", "_"), None)
        if cli_val is not None:
            cfg[k] = cli_val

    # Sanity check
    missing = [k for k in ("science_dir", "bias_dir", "outdir") if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required config values: {missing}")

    reduce_om_dataset(
        science_dir=Path(cfg["science_dir"]),
        bias_dir=Path(cfg["bias_dir"]),
        outdir=Path(cfg["outdir"]),
        camera_settings_txt=Path(cfg["camera_settings"]) if cfg["camera_settings"] else None,
        target_ra=cfg["target_ra"],
        target_dec=cfg["target_dec"],
        obs_lat=cfg["obs_lat"],
        obs_lon=cfg["obs_lon"],
        obs_height_m=cfg["obs_height"],
    )