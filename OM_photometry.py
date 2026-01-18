# OM Photometry


# ---OVERVIEW ---




# --- LIBRARY IMPORTS ---
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- 3RD PARTY IMPORTS --
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from tqdm import tqdm

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry




# --- DS9 regions ---
def load_ds9_circles(path: Path, subtract_one: bool = True):
    circles = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        sl = s.lower()
        if sl.startswith(("global", "image", "fk5", "icrs", "galactic")):
            continue
        if s.startswith("circle(") and ")" in s:
            parts = [p.strip() for p in s[s.find("(") + 1 : s.find(")")].split(",")]
            if len(parts) >= 3:
                x, y, r = map(float, parts[:3])
                if subtract_one:
                    x -= 1.0
                    y -= 1.0
                circles.append((x, y, r))
    return circles

def write_ds9_circles(path: Path, circles_xy_r, add_header=True):
    lines = []
    if add_header:
        lines += ["# Region file format: DS9 version 4.1", "image", "global color=green"]
    for x, y, r in circles_xy_r:
        lines.append(f"circle({x+1:.2f},{y+1:.2f},{r:.2f})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



# --- Tracking: 2D Gaussian centroid + FWHM ---

def _circle_bbox(xc, yc, r, nx, ny):
    x1 = int(max(0, np.floor(xc - r)))
    x2 = int(min(nx, np.ceil(xc + r) + 1))
    y1 = int(max(0, np.floor(yc - r)))
    y2 = int(min(ny, np.ceil(yc + r) + 1))
    return x1, x2, y1, y2

def _gauss2d_const(coords, A, mux, muy, sx, sy, C):
    x, y = coords
    gx = (x - mux) ** 2 / (2 * sx * sx)
    gy = (y - muy) ** 2 / (2 * sy * sy)
    return (C + A * np.exp(-(gx + gy))).ravel()

def gauss_centroid_and_fwhm(image, x0, y0, r, jump_px=None, min_snr=5.0, maxfev_good=4000):
    ny, nx = image.shape
    r_fit = max(7.0, 2.5 * r)
    x1, x2, y1, y2 = _circle_bbox(x0, y0, r_fit, nx, ny)
    sub = image[y1:y2, x1:x2]
    yy, xx = np.mgrid[y1:y2, x1:x2]
    msk = (xx - x0) ** 2 + (yy - y0) ** 2 <= r_fit * r_fit

    x_data = xx[msk].astype(float)
    y_data = yy[msk].astype(float)
    z_data = sub[msk].astype(float)

    # Background RMS from a thin ring
    rin = r_fit * 1.05
    rout = r_fit * 1.25
    rmask = ((xx - x0) ** 2 + (yy - y0) ** 2 >= rin * rin) & ((xx - x0) ** 2 + (yy - y0) ** 2 <= rout * rout)
    if rmask.any():
        ring = sigma_clip(sub[rmask], 5.0, masked=True).filled(np.nan)
        sky_rms = float(np.nanstd(ring))
    else:
        sky_rms = np.nan
    if not np.isfinite(sky_rms) or sky_rms <= 0:
        sky_rms = 1.0

    C0 = float(np.nanmedian(z_data))
    A0 = float(np.nanmax(z_data) - C0)

    # Fast reject if star basically invisible
    snr0 = A0 / sky_rms if sky_rms > 0 else 0.0
    if (A0 <= 0) or (snr0 < min_snr):
        return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    p0 = [A0, float(x0), float(y0), max(1.5, r_fit / 3), max(1.5, r_fit / 3), C0]
    bounds = (
        [0, x0 - r_fit, y0 - r_fit, 0.5, 0.5, -np.inf],
        [np.inf, x0 + r_fit, y0 + r_fit, 2 * r_fit, 2 * r_fit, np.inf],
    )

    try:
        A, mux, muy, sx, sy, C = curve_fit(
            _gauss2d_const, (x_data, y_data), z_data, p0=p0, bounds=bounds, maxfev=maxfev_good
        )[0]
    except Exception:
        return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    if jump_px is not None:
        if (mux - x0) ** 2 + (muy - y0) ** 2 > jump_px ** 2:
            return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    fwhm = 2.355 * float(np.sqrt(sx * sx + sy * sy))
    peak = float(A + C)
    return float(mux), float(muy), fwhm, peak, sky_rms, True



# --- Aperture photometry ---
def sky_stats_sigma_clip(image, annulus, *, sigma=3.0, maxiters=5):
    m = annulus.to_mask(method="center")
    cut = m.cutout(image)
    if cut is None:
        return np.nan, np.nan, 0
    data = cut[m.data.astype(bool)]
    data = data[np.isfinite(data)]
    if data.size < 20:
        return np.nan, np.nan, int(data.size)

    clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, masked=True)
    good = clipped.data[~clipped.mask]
    if good.size < 10:
        return np.nan, np.nan, int(good.size)

    sky_pp = np.median(good)
    sky_std = 1.4826 * np.median(np.abs(good - sky_pp))
    return float(sky_pp), float(sky_std), int(good.size)

def photometry_batch(image, xys, fwhm, *, k_ap=2.5, k_in=3.5, k_out=6.0, method="exact", require_full_annulus=True):
    ny, nx = image.shape
    # FWHM-scaled apertures (recommended)
    fwhm = float(fwhm) if np.isfinite(fwhm) else 4.0

    r_ap  = max(3.0, k_ap * fwhm)
    r_in  = max(r_ap * 1.2, k_in * fwhm)
    r_out = max(r_in + 2.0, k_out * fwhm)

    # Optional safety caps (prevents crazy values if FWHM spikes)
    r_ap  = float(np.clip(r_ap,  3.0, 20.0))
    r_in  = float(np.clip(r_in,  6.0, 40.0))
    r_out = float(np.clip(r_out, 10.0, 60.0))

    xys = np.asarray(xys, dtype=float)
    xs = xys[:, 0]
    ys = xys[:, 1]
    ok = np.isfinite(xs) & np.isfinite(ys)
    if require_full_annulus:
        ok &= (xs - r_out >= 0) & (xs + r_out < nx) & (ys - r_out >= 0) & (ys + r_out < ny)

    N = len(xys)
    flux = np.full(N, np.nan, dtype=float)

    if not np.any(ok):
        return flux, ok, r_ap, r_in, r_out

    pos = np.column_stack([xs[ok], ys[ok]])
    ap = CircularAperture(pos, r=r_ap)
    ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)

    tab = aperture_photometry(image, [ap, ann], method=method)
    ap_sum = np.asarray(tab["aperture_sum_0"], dtype=float)

    # Compute sky per-star properly (sigma-clipped annulus)
    flux_ok = np.full_like(ap_sum, np.nan, dtype=float)
    for i in range(len(ap_sum)):
        sky_pp, _, _ = sky_stats_sigma_clip(image, CircularAnnulus(pos[i], r_in=r_in, r_out=r_out))
        flux_ok[i] = ap_sum[i] - sky_pp * ap.area

    flux[ok] = flux_ok
    return flux, ok, r_ap, r_in, r_out



# --- Time axis ---
def choose_time_axis(manifest_csv: Path, n_frames: int):
    if not manifest_csv.exists():
        return np.arange(n_frames), "Frame index"

    df = pd.read_csv(manifest_csv).iloc[:n_frames].copy()
    for col, label in [("bjd_tdb", "BJD_TDB"), ("BJD_TDB", "BJD_TDB"), ("jd_utc", "JD_UTC"), ("JD_UTC", "JD_UTC")]:
        if col in df.columns:
            t = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isfinite(t).sum() >= max(10, int(0.2 * len(t))):
                return t, label

    return np.arange(n_frames), "Frame index"

def rank_comps_by_rms(comp_flux, base_mask, *, min_points=50):
    """
    Leave-one-out: for each comp j, form lc_j = comp_j / median(other_comps),
    normalise by OOT median (here: median of base_mask), then RMS.
    Returns rms (J,), npts (J,)
    """
    comp_flux = np.asarray(comp_flux, float)   # (N, J)
    base = np.asarray(base_mask, bool)         # (N,)

    N, J = comp_flux.shape
    rms = np.full(J, np.inf, float)
    npts = np.zeros(J, int)

    for j in range(J):
        y = comp_flux[:, j]

        others = np.delete(comp_flux, j, axis=1)
        ens_oth = np.nanmedian(others, axis=1)

        ok = base & np.isfinite(y) & np.isfinite(ens_oth) & (y > 0) & (ens_oth > 0)
        n = int(ok.sum())
        npts[j] = n
        if n < min_points:
            continue

        lc = y[ok] / ens_oth[ok]
        med = np.nanmedian(lc)
        if not np.isfinite(med) or med <= 0:
            continue
        lc = lc / med

        rms[j] = float(np.nanstd(lc))

    return rms, npts

def build_weighted_ensemble(comp_flux, idx_keep, base_mask, *, eps=1e-8):
    """
    Build ensemble from selected comps using weights ~ 1/RMS^2 (computed on base_mask).
    Returns ens (N,), weights (k,)
    """
    comps = np.asarray(comp_flux[:, idx_keep], float)  # (N, k)
    base = np.asarray(base_mask, bool)

    # normalise each comp by its median on base
    med = np.nanmedian(comps[base, :], axis=0)
    med = np.where(np.isfinite(med) & (med > 0), med, np.nan)
    rel = comps / med[None, :]

    # weights from scatter on base
    sig = np.nanstd(rel[base, :], axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    w = 1.0 / (sig * sig + eps)
    w[~np.isfinite(w)] = 0.0

    num = np.nansum(rel * w[None, :], axis=1)
    den = np.nansum(np.isfinite(rel) * w[None, :], axis=1)
    ens = num / den
    return ens, w

def decorrelate_xy_using_comps(lc_norm, targ_x, targ_y, comp_flux, comp_x, comp_y, good_mask,
                               use_fwhm=None, fwhm=None):
    """
    Fit a multiplicative trend vs x,y (and optional fwhm) using COMPARISON stars only.
    Apply the learned correction to the target lc_norm.

    lc_norm: (N,) target/ensemble normalised LC (around ~1)
    comp_flux: (N,J) sky-sub fluxes for comps
    comp_x, comp_y: (N,J) tracked positions for comps
    good_mask: (N,) boolean frames to use

    Returns: lc_corr, trend_targ, coeffs
    """
    good = np.asarray(good_mask, bool)

    # -- Build training set from comps --
    # For each comp j: make its own differential LC against OTHER comps (leave-one-out)
    N, J = comp_flux.shape
    rows_X = []
    rows_y = []

    for j in range(J):
        yj = comp_flux[:, j]
        others = np.delete(comp_flux, j, axis=1)
        ens = np.nanmedian(others, axis=1)

        ok = good & np.isfinite(yj) & np.isfinite(ens) & (yj > 0) & (ens > 0)
        if ok.sum() < 30:
            continue

        lcj = yj[ok] / ens[ok]
        med = np.nanmedian(lcj)
        if not np.isfinite(med) or med <= 0:
            continue
        lcj = lcj / med  # around 1

        xj = comp_x[ok, j]
        yjpos = comp_y[ok, j]

        ok2 = np.isfinite(xj) & np.isfinite(yjpos) & np.isfinite(lcj) & (lcj > 0)
        if ok2.sum() < 30:
            continue

        # Design matrix for multiplicative trend in log-space:
        # log(lc) ≈ c0 + c1*(x-x0) + c2*(y-y0) (+ c3*(fwhm-f0))
        x0 = np.nanmedian(xj[ok2])
        y0 = np.nanmedian(yjpos[ok2])

        X = np.column_stack([
            np.ones(ok2.sum()),
            (xj[ok2] - x0),
            (yjpos[ok2] - y0),
        ])

        if use_fwhm and (fwhm is not None):
            ff = fwhm[ok][ok2]
            f0 = np.nanmedian(ff)
            X = np.column_stack([X, (ff - f0)])

        rows_X.append(X)
        rows_y.append(np.log(lcj[ok2]))

    if not rows_X:
        # Nothing to fit; return original
        return lc_norm, np.ones_like(lc_norm), None

    Xtr = np.vstack(rows_X)
    ytr = np.concatenate(rows_y)

    # Robust-ish: plain least squares in log space
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    # -- Build target trend and correct --
    ok_t = good & np.isfinite(lc_norm) & (lc_norm > 0) & np.isfinite(targ_x) & np.isfinite(targ_y)

    x0t = np.nanmedian(targ_x[ok_t])
    y0t = np.nanmedian(targ_y[ok_t])

    Xt = np.column_stack([
        np.ones(ok_t.sum()),
        (targ_x[ok_t] - x0t),
        (targ_y[ok_t] - y0t),
    ])

    if use_fwhm and (fwhm is not None):
        ff = fwhm[ok_t]
        f0 = np.nanmedian(ff)
        Xt = np.column_stack([Xt, (ff - f0)])

    log_trend = Xt @ coef
    trend = np.ones_like(lc_norm, dtype=float)
    trend[ok_t] = np.exp(log_trend)

    lc_corr = lc_norm / trend
    
    # Re-normalise on good frames
    lc_corr = lc_corr / np.nanmedian(lc_corr[ok_t])

    return lc_corr, trend, coef

def rotate_linear_log(t, y, mask):
    """
    Remove global linear trend in log space, with centered time for numerical stability.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.asarray(mask, bool) & np.isfinite(t) & np.isfinite(y) & (y > 0)

    t0 = np.nanmedian(t[m])
    dt = t[m] - t0

    z = np.log(y[m])
    A = np.vstack([np.ones(m.sum()), dt]).T
    a, b = np.linalg.lstsq(A, z, rcond=None)[0]

    # Baseline for all points
    dt_all = t - t0
    base = np.exp(a + b * dt_all)

    y_corr = y / base
    y_corr = y_corr / np.nanmedian(y_corr[m])
    return y_corr, base, (a, b, t0)

    

# --- MAIN ---
def main(cube_path: Path, reg_path: Path, outdir: Path, *,
         k_ap=2.5, k_in=3.5, k_out=6.0,
         min_snr_ref=5.0, min_snr_targ=5.0, min_snr_comp=4.0,
         min_valid_comps=3,
         fwhm_min=0.8, fwhm_max=20.0,
         write_tracked_reg_every=25):

    outdir.mkdir(parents=True, exist_ok=True)

    cube = fits.getdata(cube_path, memmap=False)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube (N, ny, nx). Got {cube.shape}")

    N, ny, nx = cube.shape
    print(f"Loaded cube: N={N}, shape=({ny},{nx})")

    regs = load_ds9_circles(reg_path, subtract_one=True)
    if len(regs) < 2:
        raise ValueError("Need >=2 DS9 circles: target + >=1 comp")

    target_pos = (regs[0][0], regs[0][1])
    target_r = float(regs[0][2])

    comps_pos = [(x, y) for (x, y, _r) in regs[1:]]
    comps_r   = [float(r) for (_x, _y, r) in regs[1:]]

    # Use first comp as reference star for drift
    ref_pos = comps_pos[0]
    ref_r   = comps_r[0]

    # Time axis from manifest next to cube or inside outdir
    manifest = None
    for cand in [outdir / "manifest.csv", cube_path.with_name("manifest.csv"), outdir / "cube_manifest.csv", cube_path.with_name("cube_manifest.csv")]:
        if cand.exists():
            manifest = cand
            break
    if manifest is None:
        manifest = outdir / "manifest.csv"

    t, tlabel = choose_time_axis(manifest, N)

    J = len(comps_pos)
    targ_flux = np.full(N, np.nan)
    comp_flux = np.full((N, J), np.nan)
    fwhm_list = np.full(N, np.nan)
    
    # Store centroids
    targ_x = np.full(N, np.nan)
    targ_y = np.full(N, np.nan)
    
    comp_x = np.full((N, J), np.nan)
    comp_y = np.full((N, J), np.nan)

    ok_ref  = np.zeros(N, dtype=bool)
    ok_targ = np.zeros(N, dtype=bool)
    ok_comps= np.zeros(N, dtype=bool)

    fwhm_prev = 4.0

    for i in tqdm(range(N), desc="Tracked photometry", ncols=90):
        img = cube[i].astype(np.float32)

    # -- GLOBAL SKY SUBTRACTION (additive drift removal) --
        sky = np.nanmedian(img)
        img = img - sky
    
        # 1.) Fit reference
        Jref = max(0.6 * ref_r, 6.0)
        rx, ry, rfwhm, *_ = gauss_centroid_and_fwhm(img, ref_pos[0], ref_pos[1], ref_r, jump_px=Jref, min_snr=min_snr_ref)
        if not np.isfinite(rfwhm) or not np.isfinite(rx) or not np.isfinite(ry):
            continue
        ok_ref[i] = True

        # Drift update
        dx = rx - ref_pos[0]
        dy = ry - ref_pos[1]
        ref_pos = (rx, ry)

        # Fwhm update
        if np.isfinite(rfwhm) and (fwhm_min <= rfwhm <= fwhm_max):
            fwhm_prev = float(rfwhm)
        fwhm = fwhm_prev
        fwhm_list[i] = fwhm

        # 2.) Predict + fit target
        target_pred = (target_pos[0] + dx, target_pos[1] + dy)
        Jtarg = max(0.6 * target_r, 6.0)
        tx, ty, tfwhm, *_ = gauss_centroid_and_fwhm(img, target_pred[0], target_pred[1], target_r, jump_px=Jtarg, min_snr=min_snr_targ)
        if not np.isfinite(tx) or not np.isfinite(ty):
            continue
        ok_targ[i] = True
        target_pos = (tx, ty)

        # 3.) Comps: just drift-predict + optional refit each comp (lightweight)
        comps_pred = [(x + dx, y + dy) for (x, y) in comps_pos]
        new_comps = []
        for (px, py), rr in zip(comps_pred, comps_r):
            Jc = max(0.6 * rr, 6.0)
            cx, cy, cfwhm, *_ = gauss_centroid_and_fwhm(img, px, py, rr, jump_px=Jc, min_snr=min_snr_comp)
            if np.isfinite(cx) and np.isfinite(cy):
                new_comps.append((cx, cy))
            else:
                new_comps.append((px, py))
        comps_pos = new_comps

        targ_x[i], targ_y[i] = tx, ty
        for j, (cx, cy) in enumerate(comps_pos):
            comp_x[i, j] = cx
            comp_y[i, j] = cy
            
        # 4.) One batched photometry call (target + comps)
        xys_all = [(tx, ty)] + list(comps_pos)
        flux_all, ok_all, r_ap, r_in, r_out = photometry_batch(img, xys_all, fwhm, k_ap=k_ap, k_in=k_in, k_out=k_out)

        tflux = flux_all[0]
        cflux = flux_all[1:]

        # QC: require at least some valid comps
        n_ok = int(np.sum(np.isfinite(cflux) & (cflux > 0)))
        if not (np.isfinite(tflux) and tflux > 0 and n_ok >= min_valid_comps):
            continue

        ok_comps[i] = True
        targ_flux[i] = tflux
        comp_flux[i, :] = cflux

        # Tracked region overlay occasionally
        if write_tracked_reg_every and (i % int(write_tracked_reg_every) == 0):
            regs_now = [(tx, ty, target_r)] + [(x, y, r) for (x, y), r in zip(comps_pos, comps_r)]
            write_ds9_circles(outdir / f"tracked_{i:04d}.reg", regs_now)



    # --- Ensemble + LC --- (RMS-selected comps)
    # Base "good frame" mask BEFORE dividing by ensemble
    base = ok_ref & ok_targ & ok_comps & np.isfinite(targ_flux) & (targ_flux > 0) & np.isfinite(t)
    
    # Rank comps by leave-one-out RMS
    rms, npts = rank_comps_by_rms(comp_flux, base, min_points=max(30, int(0.3 * base.sum())))
    
    # Choose best k comps
    J = comp_flux.shape[1]
    k_best = min(12, J)
    idx_sorted = np.argsort(rms)
    idx_keep = [int(j) for j in idx_sorted if np.isfinite(rms[j])][:k_best]
    
    # Fallback: if something went wrong, just use all comps
    if len(idx_keep) < 2:
        idx_keep = list(range(J))
    
    # Write ranking report
    rank_path = outdir / "comp_rank.txt"
    with rank_path.open("w", encoding="utf-8") as f:
        f.write("Comparison star ranking (lower RMS is better)\n")
        f.write(f"Selected k_best={len(idx_keep)} of J={J}\n\n")
        f.write("rank  comp_idx   RMS        Npts\n")
        for r, j in enumerate(idx_sorted[:min(J, 30)], start=1):
            f.write(f"{r:>4d}  {j:>8d}  {rms[j]:>8.6f}  {npts[j]:>6d}\n")
        f.write("\nKEEP:\n")
        for j in idx_keep:
            f.write(f"  comp[{j}]  RMS={rms[j]:.6f}  Npts={npts[j]}\n")
    
    print(f"\nBest comps kept: {idx_keep}")
    print(f"Saved comp ranking: {rank_path}")
    
    # Ensemble choice: "median" is often more stable under colour/systematics than weighted
    ENSEMBLE_MODE = "median"   # "median" or "weighted"
    
    if ENSEMBLE_MODE == "weighted":
        ens, w = build_weighted_ensemble(comp_flux, idx_keep, base)
    else:
        comps_sel = comp_flux[:, idx_keep]
        
        # Normalise each comp by its median on base, then take median across comps
        med = np.nanmedian(comps_sel[base, :], axis=0)
        med = np.where(np.isfinite(med) & (med > 0), med, np.nan)
        rel = comps_sel / med[None, :]
        ens = np.nanmedian(rel, axis=1)
    
    # Differential LC
    lc = targ_flux / ens
    
    good = base & np.isfinite(lc) & (lc > 0) & np.isfinite(ens) & (ens > 0)
    print(f"Frames: N={N} | good={int(good.sum())} ({100*np.mean(good):.1f}%)")
    if good.sum() < 10:
        raise SystemExit("❌ Still basically no valid points after RMS selection.")
    
    lc_norm = lc / np.nanmedian(lc[good])


    # --- PRECOMPUTE xy model coefficients from comps once (for injection–recovery) ---
    lc_xycorr, trend, coef = decorrelate_xy_using_comps(
        lc_norm,
        targ_x, targ_y,
        comp_flux, comp_x, comp_y,
        good_mask=good,
        use_fwhm=True,
        fwhm=fwhm_list
    )

    # Trend model diagnostic (what gets divided out)
    plt.figure(figsize=(10, 3.5))
    plt.plot(t[good], trend[good], ".", ms=2.5)
    plt.xlabel(tlabel)
    plt.ylabel("Multiplicative trend")
    plt.title("XY(+FWHM) decorrelation trend (comparison-trained)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "om_xy_trend.png", dpi=300)
    plt.close()

    def apply_xycoef_to_target(lc_norm_in):
        """
        Apply pre-fit xy(+fwhm) coefficients (coef) to a *target* lc_norm_in.
        This avoids refitting the comp model for every injection trial.
        """
        if coef is None:
            return lc_norm_in

        lc_norm_in = np.asarray(lc_norm_in, float)
        ok_t = good & np.isfinite(lc_norm_in) & (lc_norm_in > 0) & np.isfinite(targ_x) & np.isfinite(targ_y)

        x0t = np.nanmedian(targ_x[ok_t])
        y0t = np.nanmedian(targ_y[ok_t])

        Xt = np.column_stack([
            np.ones(ok_t.sum()),
            (targ_x[ok_t] - x0t),
            (targ_y[ok_t] - y0t),
            (fwhm_list[ok_t] - np.nanmedian(fwhm_list[ok_t])),
        ])

        log_trend = Xt @ coef
        trend_t = np.ones_like(lc_norm_in, dtype=float)
        trend_t[ok_t] = np.exp(log_trend)

        lc_corr = lc_norm_in / trend_t
        lc_corr = lc_corr / np.nanmedian(lc_corr[ok_t])
        return lc_corr
    
    # Save an extra plot / CSV column
    lc_norm_raw = lc_norm.copy()
    lc_norm = lc_xycorr
    lc_rot_raw, _, _ = rotate_linear_log(t, lc_norm_raw, good)  # rotate the raw LC
    lc_rot_xy,  _, _ = rotate_linear_log(t, lc_norm,     good)  # rotate the xy-corrected LC


    # - Thesis figure: processing ladder (raw -> xy-corrected -> rotated) -
    def _robust_ylim(y, mask, pad_frac=0.15):
        yy = np.asarray(y, float)[mask]
        yy = yy[np.isfinite(yy)]
        if yy.size < 10:
            return None
        lo, hi = np.nanpercentile(yy, [1, 99])
        if not (np.isfinite(lo) and np.isfinite(hi)) or (hi <= lo):
            return None
        pad = pad_frac * (hi - lo)
        return (lo - pad, hi + pad)

    ylim = _robust_ylim(lc_norm_raw, good, pad_frac=0.20)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    axes[0].plot(t[good], lc_norm_raw[good], ".", ms=2.5)
    axes[0].set_ylabel("Rel. flux")
    axes[0].set_title("Raw differential LC (normalised)")

    axes[1].plot(t[good], lc_norm[good], ".", ms=2.5)
    axes[1].set_ylabel("Rel. flux")
    axes[1].set_title("After x/y(+FWHM) decorrelation (normalised)")

    axes[2].plot(t[good], lc_rot_xy[good], ".", ms=2.5)
    axes[2].set_ylabel("Rel. flux")
    axes[2].set_title("After log-linear rotation (final)")
    axes[2].set_xlabel(tlabel)

    if ylim is not None:
        for ax in axes:
            ax.set_ylim(*ylim)

    for ax in axes:
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outdir / "om_processing_ladder.png", dpi=300)
    plt.close()
    

    # --- Injection–recovery on lc_rot_xy (OM control) ---
    def mad_std(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            return np.nan
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med))

    def inject_box_transit(t, y, t0, dur_days, depth):
        """
        Multiplicative box transit injection:
        y_inj = y * (1 - depth) in-transit, else y unchanged.
        depth is fractional (e.g. 0.005 = 0.5%).
        """
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        in_tr = np.abs(t - t0) <= 0.5 * dur_days
        y_inj = y.copy()
        y_inj[in_tr] *= (1.0 - depth)
        return y_inj, in_tr

    def sliding_box_search(t, y, dur_days, local_mult=3.0):
        """
        (A) Sliding-box matched filter with (B) local baseline.
        For each candidate center time, compute depth_hat and SNR
        using IN window +/-dur/2 and a local OOT region within
        +/- local_mult*dur excluding the in-window.

        Returns best_t0, best_depth_hat, best_snr
        """
        t = np.asarray(t, float)
        y = np.asarray(y, float)

        ok = np.isfinite(t) & np.isfinite(y)
        t = t[ok]
        y = y[ok]

        if t.size < 30:
            return np.nan, np.nan, np.nan

        best_snr = -np.inf
        best_t0 = np.nan
        best_d = np.nan

        half = 0.5 * dur_days
        span = local_mult * dur_days

        for t0 in t:
            in_tr = np.abs(t - t0) <= half
            if np.sum(in_tr) < 5:
                continue

            # Local baseline region
            local = np.abs(t - t0) <= span
            oot = local & (~in_tr)
            if np.sum(oot) < 10:
                continue

            med_in = np.nanmedian(y[in_tr])
            med_oot = np.nanmedian(y[oot])
            depth_hat = med_oot - med_in  # positive dip

            sig = mad_std(y[oot])
            if not np.isfinite(sig) or sig <= 0:
                continue

            # Std error of difference of means-ish (stable detection statistic)
            n_in = np.sum(in_tr)
            n_oot = np.sum(oot)
            se = sig * np.sqrt(1.0 / n_in + 1.0 / n_oot)
            if se <= 0:
                continue

            snr = depth_hat / se
            if np.isfinite(snr) and snr > best_snr:
                best_snr = snr
                best_t0 = t0
                best_d = depth_hat

        if not np.isfinite(best_snr):
            return np.nan, np.nan, np.nan
        return best_t0, best_d, best_snr

    idx_good = np.where(good)[0]
    t_use = np.asarray(t[idx_good], float)
    y_use = np.asarray(lc_rot_xy[idx_good], float)
    
    m = np.isfinite(t_use) & np.isfinite(y_use)
    idx_use = idx_good[m]
    t_use = t_use[m]
    y_use = y_use[m]

    if t_use.size >= 50:
        tmin, tmax = np.nanmin(t_use), np.nanmax(t_use)
        ts = np.sort(t_use)
        dt_med = np.nanmedian(np.diff(ts))
        if not np.isfinite(dt_med) or dt_med <= 0:
            dt_med = (tmax - tmin) / max(1, (t_use.size - 1))

        # --- SETTINGS ---
        dur_frames = 30                      # 20, 30, 40 depending on cadence
        dur_days = dur_frames * dt_med
        depths_frac = np.array([0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020])  # 0.2%..2.0%
        n_trials = 50
        snr_thresh = 4.0
        local_mult = 3.0                     # (B) local baseline extent = ±(local_mult*dur)
        pad = 1.2 * dur_days                 # keep injections away from edges
        # --------------

        rng = np.random.default_rng(42)
        rows = []

        for d in depths_frac:
            n_ok = 0
            snrs = []
            dhats = []

            for _ in range(n_trials):
                if (tmax - tmin) <= 2 * pad:
                    break

                t0_inj = rng.uniform(tmin + pad, tmax - pad)

                # 1.) Inject into the PRE-detrended series
                y0 = lc_norm_raw[idx_use].copy()
                y_inj, _ = inject_box_transit(t_use, y0, t0_inj, dur_days, d)
                
                # 2.) Put injected series back into full-length array so apply_xycoef_to_target() can use good mask
                tmp = lc_norm_raw.copy()
                tmp[idx_use] = y_inj
                
                # 3.) Apply xy(+fwhm) correction and then rotate
                tmp_xy = apply_xycoef_to_target(tmp)
                tmp_rot, _, _ = rotate_linear_log(t, tmp_xy, good)
                
                # 4.) Detect on the same product you'd use for science
                y_det = tmp_rot[good]
                t_det = t[good]
                t0_best, d_hat, snr = sliding_box_search(t_det, y_det, dur_days, local_mult=local_mult)
                
                # Recovery criteria:
                #   - detection at threshold
                #   - best time close to injected time
                #   - positive depth, roughly consistent (factor of 2)
                ok = (
                    np.isfinite(snr) and (snr >= snr_thresh) and
                    np.isfinite(t0_best) and (np.abs(t0_best - t0_inj) <= 0.5 * dur_days) and
                    np.isfinite(d_hat) and (d_hat > 0) and (0.5 * d <= d_hat <= 2.0 * d)
                )

                if ok:
                    n_ok += 1
                if np.isfinite(snr):
                    snrs.append(float(snr))
                if np.isfinite(d_hat):
                    dhats.append(float(d_hat))

            frac = n_ok / n_trials if n_trials > 0 else np.nan
            rows.append({
                "depth_frac": float(d),
                "depth_percent": float(100.0 * d),
                "dur_frames": int(dur_frames),
                "dur_days": float(dur_days),
                "n_trials": int(n_trials),
                "recovered": int(n_ok),
                "recovery_frac": float(frac),
                "snr_median": float(np.nanmedian(snrs)) if snrs else np.nan,
                "depthhat_median_frac": float(np.nanmedian(dhats)) if dhats else np.nan,
            })

        df_ir = pd.DataFrame(rows)
        df_ir.to_csv(outdir / "om_injection_recovery.csv", index=False)

        plt.figure(figsize=(6, 4))
        plt.plot(df_ir["depth_percent"], df_ir["recovery_frac"], "o-")
        plt.axhline(0.5, ls="--", lw=1)
        plt.axhline(0.9, ls="--", lw=1)
        plt.xlabel("Injected depth (%)")
        plt.ylabel("Recovery fraction")
        plt.title(f"OM injection–recovery /n SNR≥{snr_thresh:.1f}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "om_injection_recovery_efficiency.png", dpi=300)
        plt.close()

        print("✅ Injection–recovery saved:",
              outdir / "om_injection_recovery.csv",
              outdir / "om_injection_recovery_efficiency.png")
    else:
        print("⚠️ Skipping injection–recovery: too few finite points in lc_rot_xy.")

        
        
    # --- Noise diagnostics (OM control): RMS vs bin size, beta, autocorr ---
    def binned_scatter_of_means(resid, bin_sizes):
        resid = np.asarray(resid, float)
        resid = resid[np.isfinite(resid)]
        out = np.full(len(bin_sizes), np.nan, float)
        for i, N in enumerate(bin_sizes):
            N = int(N)
            if N <= 0:
                continue
            n_bins = resid.size // N
            
            # Require at least 10 independent bins
            if n_bins < 10:   
                continue
            n_full = n_bins * N
            r = resid[:n_full].reshape(n_bins, N)
            m = np.nanmean(r, axis=1)
            out[i] = np.nanstd(m, ddof=1)
        return out

    def autocorr(x, max_lag=200):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        x = x - np.nanmean(x)
        if x.size < 10:
            return None, None
        max_lag = min(int(max_lag), x.size - 2)
        ac = np.correlate(x, x, mode="full")[x.size-1:x.size+max_lag]
        ac /= ac[0]
        lags = np.arange(0, max_lag+1)
        return lags, ac

    flux_use = lc_rot_xy
    resid = flux_use[good] - np.nanmedian(flux_use[good])

    # -- Systematics check: residuals vs x, y, FWHM --
    xg = targ_x[good]
    yg = targ_y[good]
    fg = fwhm_list[good]
    rg = resid

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    axes[0].plot(xg, rg, ".", ms=2.5)
    axes[0].axhline(0.0, lw=1)
    axes[0].set_xlabel("Target x (px)")
    axes[0].set_ylabel("Residuals")

    axes[1].plot(yg, rg, ".", ms=2.5)
    axes[1].axhline(0.0, lw=1)
    axes[1].set_xlabel("Target y (px)")

    axes[2].plot(fg, rg, ".", ms=2.5)
    axes[2].axhline(0.0, lw=1)
    axes[2].set_xlabel("FWHM (px)")

    for ax in axes:
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outdir / "om_resid_vs_xy_fwhm.png", dpi=300)
    plt.close()
    bin_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
    sigN = binned_scatter_of_means(resid, bin_sizes)

    # Guard against NaNs
    if np.isfinite(sigN[0]) and sigN[0] > 0:
        sig1 = sigN[0]
        white = sig1 / np.sqrt(bin_sizes)
        beta = sigN / white

        # RMS vs bin size
        plt.figure(figsize=(6, 4))
        plt.loglog(bin_sizes, sigN, "o-", label="Binned scatter")
        plt.loglog(bin_sizes, white, "k--", label=r"White noise $\propto 1/\sqrt{N}$")
        plt.xlabel("Bin size (points)")
        plt.ylabel("Scatter of binned means")
        plt.title("OM time-averaging (raw rotated)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "om_noise_rms_vs_bins.png", dpi=300)
        plt.close()

        # Beta vs bin size
        plt.figure(figsize=(6, 4))
        plt.semilogx(bin_sizes[1:], beta[1:], "o-")
        plt.axhline(1.0, ls="--", c="k", lw=1)
        plt.xlabel("Bin size (points)")
        plt.ylabel(r"$\beta(N)$")
        plt.title("OM red-noise factor (raw rotated)")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "om_noise_beta_vs_bins.png", dpi=300)
        plt.close()

        # Autocorrelation
        lags, ac = autocorr(resid, max_lag=200)
        if lags is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(lags, ac, "o-")
            plt.axhline(0.0, c="k", lw=1)
            plt.xlabel("Lag (frames)")
            plt.ylabel("Autocorrelation")
            plt.title("OM residual autocorrelation (raw rotated)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "om_resid_autocorr.png", dpi=300)
            plt.close()
    else:
        print("⚠️ Skipping OM noise diagnostics: could not compute finite sig1.")
        
    # Save
    out_csv = outdir / f"lc_tracked_{cube_path.stem}.csv"
    pd.DataFrame({
        "t": t,
        "lc_norm_raw": lc_norm_raw,
        "lc_norm_xycorr": lc_norm,
        "targ_x": targ_x,
        "targ_y": targ_y,
        "targ_flux": targ_flux,
        "ens_flux": ens,
        "ok": good.astype(int),
        "fwhm": fwhm_list,
    }).to_csv(out_csv, index=False)

    
    
    # - Plot -
    out_png_xy  = outdir / f"lc_xy_{cube_path.stem}.png"
    out_png_rot = outdir / f"lc_rot_{cube_path.stem}.png"

    # RAW vs XY
    plt.figure(figsize=(10,4))
    plt.plot(t[good], lc_norm_raw[good], ".", ms=3, label="raw")
    plt.xlabel(tlabel); plt.ylabel("Relative flux (norm)")
    plt.title("OM LC")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png_xy, dpi=160); plt.close()
    
    # ROTATED versions
    plt.figure(figsize=(10,4))
    plt.plot(t[good], lc_rot_raw[good], ".", ms=3, label="raw rotated")
    plt.plot(t[good], lc_rot_xy[good], ".", ms=3, label="xy-corrected rotated")
    plt.xlabel(tlabel); plt.ylabel("Relative flux (norm)")
    plt.title("OM LC: log-linear rotation")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png_rot, dpi=160); plt.close()

    def binom_sigma(p, n):
        p = np.asarray(p, float)
        n = np.asarray(n, float)
        return np.sqrt(np.clip(p * (1 - p) / np.maximum(n, 1.0), 0, np.inf))
    
    # Compute 1-sigma binomial errors
    p = df_ir["recovery_frac"].to_numpy()
    n = df_ir["n_trials"].to_numpy()
    sig = binom_sigma(p, n)
    
    # Find 50% and 90% completeness depths (simple linear interp)
    def completeness_depth(depths, frac, level):
        depths = np.asarray(depths, float)
        frac = np.asarray(frac, float)
        ok = np.isfinite(depths) & np.isfinite(frac)
        depths, frac = depths[ok], frac[ok]
        if depths.size < 2:
            return np.nan
        
        # Require monotonic-ish behaviour; just find first crossing
        idx = np.where(frac >= level)[0]
        if idx.size == 0:
            return np.nan
        i = idx[0]
        if i == 0:
            return depths[0]
        
        # Linear interpolation between i-1 and i
        x0, x1 = depths[i-1], depths[i]
        y0, y1 = frac[i-1], frac[i]
        if y1 == y0:
            return x1
        return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    
    d50 = completeness_depth(df_ir["depth_percent"], df_ir["recovery_frac"], 0.5)
    d90 = completeness_depth(df_ir["depth_percent"], df_ir["recovery_frac"], 0.9)
    
    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(df_ir["depth_percent"], df_ir["recovery_frac"], yerr=sig, fmt="o-", capsize=3)
    
    # Reference lines
    plt.axhline(0.5, ls="--", lw=1)
    plt.axhline(0.9, ls="--", lw=1)
    
    # Vertical completeness markers
    if np.isfinite(d50):
        plt.axvline(d50, ls="--", lw=1)
        plt.text(d50, 0.52, f"50% at {d50:.2f}%", rotation=0, fontsize=10, va="bottom", ha="right")
    if np.isfinite(d90):
        plt.axvline(d90, ls="--", lw=1)
        plt.text(d90, 0.92, f"90% at {d90:.2f}%", rotation=0, va="bottom", ha="right")
    else:
        plt.text(df_ir["depth_percent"].max(), 0.92, "90% not reached", ha="right", va="bottom")
    
    plt.ylim(-0.02, 1.02)
    plt.xlim(df_ir["depth_percent"].min() - 0.05, df_ir["depth_percent"].max() + 0.05)
    plt.xlabel("Injected depth (%)")
    plt.ylabel("Recovery fraction")
    plt.title(f"OM injection–recovery SNR≥{snr_thresh:.1f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "om_injection_recovery_efficiency_pretty.png", dpi=300)
    plt.close()
    
if __name__ == "__main__":
    
    # - EDIT ONCE -
    CUBE = Path(r"C:\Users\Jamie McAteer\PythonSaves\Fits\TOI_4546_b\reduction\OM_green_cube.fits")
    REG  = Path(r"C:\Users\Jamie McAteer\PythonSaves\Fits\TOI_4546_b\regions\om_regions.reg")
    OUT  = Path(r"C:\Users\Jamie McAteer\PythonSaves\Fits\TOI_4546_b\photometry_tracked_min")

    # Start with forgiving settings
    main(
        CUBE, REG, OUT,
        k_ap=2.5, k_in=3.5, k_out=6.0,
        min_snr_ref=4.0, min_snr_targ=4.0, min_snr_comp=3.5,
        min_valid_comps=2,
        write_tracked_reg_every=25
    )
    