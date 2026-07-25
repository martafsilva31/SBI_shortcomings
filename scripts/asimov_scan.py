"""Compute engine for the Asimov vs median-of-toys reproduction (redo of 2026-07-16).

Produces the raw likelihood scans that notebooks/histogram_reproduction.ipynb and
notebooks/nsbi_reproduction.ipynb interpret. Also importable as a module: the notebooks
take the statistic definitions from here so there is exactly one implementation.

Definitions (plain terms; asymptotic references are Cowan-Cranmer-Gross-Vitells,
arXiv:1007.1727)
----------------------------------------------------------------------------------------
For a dataset with per-event counts k_i and a model predicting f_i(mu) events at event i,
the extended likelihood treats every event as its own Poisson bin:

    -2 ln L(mu) = 2 * sum_i [ f_i(mu) - k_i * ln f_i(mu) ]   (+ terms independent of mu)

The histogram analysis is the same expression with events grouped into m4l bins.
The per-toy test statistic is

    t(mu) = -2 [ ln L(mu) - ln L(mu_hat) ],   mu_hat = argmax of that toy's likelihood,

computed on the mu grid. In the large-sample limit, t(mu) for toys generated at mu_true
follows a noncentral chi-square distribution with 1 degree of freedom whose
noncentrality parameter is the Asimov value t_A(mu); the functions ncx2_cdf/ncx2_quantile
below implement that reference distribution.

Analyses
--------
hist        m4l histogram (10 log bins, [180, 1000] GeV), per-bin Poisson likelihood
            (stats/binned.py, unchanged).
nsbi-exact  unbinned likelihood with the EXACT per-event ratio built from the stored
            MCFM matrix elements (an NSBI analysis with a perfect, error-free ratio).
nsbi-nn     the same unbinned likelihood with the ratio estimated by Tae's two CARL
            networks (checkpoints from his CERNBox share).

--shape-only reproduces Tae's committed statistic (no event-count term); it exists to
quantify that omission, not as a target.
"""

import argparse
import glob
import json
import os
import pickle
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
SUBREPO = os.path.join(ROOT, "asimov-aymptotics-check")


# ---------------------------------------------------------------------------------------
# CERNBox share download (step C0). Standard library only, so this part runs on nodes
# without the container: when called with --download-checkpoints the script exits before
# the analysis imports (numpy/torch/sub-repo) below.
# ---------------------------------------------------------------------------------------
def download_share(token, out_dir, include_train_pkls=False):
    """Mirror the public CERNBox share into out_dir over anonymous WebDAV, skipping the
    training splits unless asked. Existing files with the right size are not re-fetched."""
    base = f"https://cernbox.cern.ch/remote.php/dav/public-files/{token}"
    xml = subprocess.run(
        ["curl", "-s", "-u", f"{token}:", "-X", "PROPFIND", "-H", "Depth: infinity", base + "/"],
        capture_output=True, text=True, check=True,
    ).stdout
    entries = []
    for resp in re.findall(r"<d:response>(.*?)</d:response>", xml, re.S):
        href = re.search(r"<d:href>([^<]+)</d:href>", resp).group(1)
        size = re.search(r"<d:getcontentlength>(\d+)</d:getcontentlength>", resp)
        if size is None:
            continue  # a directory
        rel = href.split(f"/public-files/{token}/", 1)[1]
        rel = rel.replace("%3d", "=").replace("%3D", "=")
        entries.append((rel, int(size.group(1))))
    if not entries:
        raise RuntimeError("empty WebDAV listing — check the share token / connectivity")

    skipped, fetched = [], []
    for rel, size in sorted(entries):
        if not include_train_pkls and re.search(r"events_(numerator|denominator)_train\.pkl$", rel):
            skipped.append(rel)
            continue
        dest = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            continue
        from urllib.parse import quote
        url = f"{base}/{quote(rel)}"
        print(f"fetching {rel}  ({size/1e6:.1f} MB)", flush=True)
        subprocess.run(["curl", "-s", "-u", f"{token}:", "-C", "-", "-o", dest, url], check=True)
        got = os.path.getsize(dest)
        if got != size:
            raise RuntimeError(f"size mismatch for {rel}: expected {size}, got {got}")
        fetched.append(rel)
    print(f"done: {len(fetched)} fetched, {len(skipped)} skipped (training splits), "
          f"{len(entries) - len(fetched) - len(skipped)} already present")


if __name__ == "__main__" and "--download-checkpoints" in sys.argv:
    _p = argparse.ArgumentParser()
    _p.add_argument("--download-checkpoints", action="store_true")
    _p.add_argument("--share-token", default="Shv8zta4oZ1aOL4")
    _p.add_argument("--include-train-pkls", action="store_true")
    _p.add_argument("--out", required=True)
    _a = _p.parse_args()
    download_share(_a.share_token, _a.out, _a.include_train_pkls)
    sys.exit(0)


import numpy as np  # noqa: E402
import torch  # noqa: E402

# path bootstrap: make the sub-repo (physics/, stats/, models/, nsbi/) importable both
# from the container root and from notebooks/
if SUBREPO not in sys.path:
    sys.path.insert(0, SUBREPO)

# LightningCLI-saved checkpoints need jsonargparse to deserialise, and it is NOT in the
# container (same gap wifi_fit.py hit). It lives in output/pyextra; it must be on the path
# before lightning is imported, or CARL.load_from_checkpoint dies with a cryptic
# "object() takes no arguments".
_PYEXTRA = os.path.join(ROOT, "output", "pyextra")
if os.path.isdir(_PYEXTRA) and _PYEXTRA not in sys.path:
    sys.path.insert(0, _PYEXTRA)

from physics.simulation import mcfm, toys  # noqa: E402
from physics.hstar import sigstr  # noqa: E402
from stats.binned import (  # noqa: E402
    poisson_log_likelihood_scan,
    poisson_log_likelihood_scan_toys,
)

FEATURES_4L = [
    "l1_pt", "l1_eta", "l1_phi", "l1_energy",
    "l2_pt", "l2_eta", "l2_phi", "l2_energy",
    "l3_pt", "l3_eta", "l3_phi", "l3_energy",
    "l4_pt", "l4_eta", "l4_phi", "l4_energy",
]
N_BINS, M4L_LO, M4L_HI = 10, 180.0, 1000.0

_TINY = 1e-300  # clamp for ln of a vanishing expected rate


# ---------------------------------------------------------------------------------------
# noncentral chi-square with 1 degree of freedom (the asymptotic reference distribution)
# ---------------------------------------------------------------------------------------
def _phi(x):
    """Standard normal CDF."""
    from math import erf
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def ncx2_cdf(t, lam):
    """P(T <= t) for T = (Z + sqrt(lam))^2, Z standard normal.

    This is the CDF of a noncentral chi-square variate with 1 degree of freedom and
    noncentrality parameter lam. At lam = 0 it is the chi-square(1 dof) CDF.
    """
    t = np.asarray(t, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    st, sl = np.sqrt(np.clip(t, 0, None)), np.sqrt(np.clip(lam, 0, None))
    return np.clip(_phi(st - sl) + _phi(st + sl) - 1.0, 0.0, 1.0)


def ncx2_quantile(p, lam):
    """Inverse of ncx2_cdf in t, by bisection. p scalar, lam scalar or array."""
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    lo = np.zeros_like(lam)
    hi = np.full_like(lam, 4.0)
    # grow hi until the CDF exceeds p everywhere (hi = lam + few sigma is plenty)
    while np.any(ncx2_cdf(hi, lam) < p):
        hi = np.where(ncx2_cdf(hi, lam) < p, hi * 2.0 + 4.0, hi)
    for _ in range(80):  # 80 halvings: |hi-lo| < 1e-20 * initial — far below any use
        mid = 0.5 * (lo + hi)
        below = ncx2_cdf(mid, lam) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = 0.5 * (lo + hi)
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------------------
# statistic utilities (imported by both notebooks — single implementation)
# ---------------------------------------------------------------------------------------
def t_curves(nll_raw, mu_grid):
    """Per-curve test statistic and best fit from raw -2 ln L curves.

    nll_raw : (n_mu,) or (n_mu, n_toys) array of -2 ln L values on the mu grid.
    Returns (t, mu_hat): t is nll_raw minus each curve's own minimum (so t >= 0 with
    t(mu_hat) = 0); mu_hat is the grid point of each curve's minimum.
    """
    a = np.asarray(nll_raw, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    idx = np.nanargmin(a, axis=0)
    t = a - a[idx, np.arange(a.shape[1])][None, :]
    mu_hat = np.asarray(mu_grid, dtype=np.float64)[idx]
    return (t.squeeze(), mu_hat if mu_hat.size > 1 else float(mu_hat[0]))


def crossing_interval(t, mu_grid, level=1.0):
    """(lo, hi, width) where the curve t crosses `level` on either side of its minimum.

    Linear interpolation between grid points; an endpoint is nan when the curve does not
    reach `level` on that side inside the scan window (interval not contained in window).
    """
    d = np.asarray(t, dtype=np.float64)
    g = np.asarray(mu_grid, dtype=np.float64)
    i = int(np.nanargmin(d))
    lo = np.interp(level, d[: i + 1][::-1], g[: i + 1][::-1]) if i > 0 and np.nanmax(d[: i + 1]) >= level else np.nan
    hi = np.interp(level, d[i:], g[i:]) if i < len(d) - 1 and np.nanmax(d[i:]) >= level else np.nan
    return lo, hi, hi - lo


def toy_quantiles(t_toys, qs=(0.16, 0.50, 0.84)):
    """Quantiles across toys at each mu grid point. t_toys: (n_mu, n_toys)."""
    return np.quantile(np.asarray(t_toys, dtype=np.float64), qs, axis=1)


def predicted_quantiles(t_asimov, qs=(0.16, 0.50, 0.84)):
    """The asymptotic prediction for toy_quantiles: quantiles of the noncentral
    chi-square(1 dof) distribution with noncentrality parameter t_A(mu)."""
    lam = np.asarray(t_asimov, dtype=np.float64)
    return np.stack([np.asarray([ncx2_quantile(q, l) for l in lam]) for q in qs])


def bootstrap_quantile_width_ratio(t_toys, t_asimov, mu_grid, n_boot=400, level=1.0, seed=1):
    """Width of the median-of-t curve (measured from its own minimum, threshold `level`
    above it) divided by the Asimov width, with a bootstrap-over-toys error bar.

    Returns (ratio, error). nan when either interval is not contained in the window.
    """
    T = np.asarray(t_toys, dtype=np.float64)
    _, _, w_a = crossing_interval(np.asarray(t_asimov) - np.min(t_asimov), mu_grid, level)

    def _width(sample):
        med = np.quantile(sample, 0.5, axis=1)
        _, _, w = crossing_interval(med - med.min(), mu_grid, level)
        return w

    r0 = _width(T) / w_a
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, T.shape[1], T.shape[1])
        vals.append(_width(T[:, pick]) / w_a)
    vals = np.asarray(vals)
    return r0, float(np.nanstd(vals))


# ---------------------------------------------------------------------------------------
# data loading and per-event rate model
# ---------------------------------------------------------------------------------------
def load_xs(data_dir):
    with open(os.path.join(data_dir, "ggzz4l_xs.json")) as f:
        xs = json.load(f)
    return {c: float(np.prod(xs[c])) for c in ("sig", "bkg", "int", "sbi")}


def load_sbi_events(data_dir, n_rows):
    """The SBI sample with weights normalised so their sum is the SBI cross-section [fb]."""
    xs = load_xs(data_dir)
    events = mcfm.from_csv(
        file_path=os.path.join(data_dir, "ggzz4l_sbi.csv"),
        kinematics=FEATURES_4L + ["4l_mass"],
        cross_section=xs["sbi"],
        n_rows=int(n_rows),
    )
    return events, xs


def unit_weights(events, mu_grid):
    """Per-event expected weights at luminosity 1 fb^-1: w_sm (N,) at mu = 1 and
    w_scan (N, n_mu) over the grid, from the stored matrix elements (sigstr.scale)."""
    w_sm, _ = sigstr.scale(events, signal_strength=1.0, ignore_interference=False)
    w_scan, _ = sigstr.scale(events, signal_strength=torch.as_tensor(np.asarray(mu_grid), dtype=torch.float64),
                             ignore_interference=False)
    return np.asarray(w_sm, dtype=np.float64), np.asarray(w_scan, dtype=np.float64)


def exact_log_density_ratio(events, xs, mu_grid, nu_sample=None):
    """ln of the exact density ratio p(x|mu)/p_bkg(x), per event and mu grid point.

    From the stored matrix elements:  r(x|mu) = xs_bkg * [mu*msq_sig + sqrt(mu)*msq_int
    + msq_bkg] / (nu(mu) * msq_bkg), with nu(mu) = mu*xs_sig + sqrt(mu)*xs_int + xs_bkg
    (cross-sections in fb; the luminosity cancels in the ratio).

    nu_sample: optional (n_mu,) sample-based yield Sum_i w_i(mu). When given, it replaces
    the cross-section-table nu(mu) in the density normalisation, making p(x|mu) exactly
    normalised on the event sample. The two differ by sub-percent Monte-Carlo differences
    — irrelevant for the full likelihood, but the shape-only statistic is flat enough in
    mu for that tilt to move its minimum by O(1) (measured in the notebook).
    """
    mu = np.asarray(mu_grid, dtype=np.float64)
    comp = events.components
    msq_sig = comp[mcfm.csv_component_sm[mcfm.Component.SIG]].to_numpy(dtype=np.float64)
    msq_int = comp[mcfm.csv_component_sm[mcfm.Component.INT]].to_numpy(dtype=np.float64)
    msq_bkg = np.clip(comp[mcfm.csv_component_sm[mcfm.Component.BKG]].to_numpy(dtype=np.float64),
                      _TINY, None)   # a handful of events underflow to msq_bkg = 0
    num = (msq_sig[:, None] * mu[None, :]
           + msq_int[:, None] * np.sqrt(mu)[None, :]
           + msq_bkg[:, None])
    nu_mu = mu * xs["sig"] + np.sqrt(mu) * xs["int"] + xs["bkg"]
    norm = nu_sample if nu_sample is not None else nu_mu
    scale = xs["bkg"] if nu_sample is None else float(np.sum(
        events.weights.to_numpy(dtype=np.float64).clip(0) * msq_bkg
        / comp[mcfm.csv_component_sm[mcfm.Component.SBI]].to_numpy(dtype=np.float64)))
    ratio = scale * num / (np.asarray(norm)[None, :] * msq_bkg[:, None])
    return np.log(np.clip(ratio, _TINY, None)), nu_mu


# ---------------------------------------------------------------------------------------
# CARL networks (nsbi-nn)
# ---------------------------------------------------------------------------------------
def load_carl(carl_dir, policy="largest-epoch"):
    """Load one CARL classifier + its input scaler from a run directory
    (layout of Tae's run/h4l/<name>/), on CPU.

    policy 'largest-epoch' replicates nsbi/carl/utils.py:load_results (Tae's choice);
    'best-val' picks the checkpoint with the smallest stored val_loss (the filename only
    carries 2 decimals, so the true score is read from the checkpoint's callback state).
    """
    from models.carl import CARL

    logs_dir = os.path.join(carl_dir, "lightning_logs")
    versions = [d for d in os.listdir(logs_dir) if re.match(r"version_\d+", d)]
    latest = max(versions, key=lambda v: int(re.search(r"\d+", v).group()))
    ckpt_dir = os.path.join(logs_dir, latest, "checkpoints")
    ckpts = [f for f in os.listdir(ckpt_dir) if re.match(r"epoch=\d+-val_loss=.+\.ckpt", f)]
    if not ckpts:
        raise FileNotFoundError(f"no val_loss checkpoints in {ckpt_dir}")

    if policy == "largest-epoch":
        chosen = max(ckpts, key=lambda f: int(re.search(r"epoch=(\d+)", f).group(1)))
    elif policy == "best-val":
        def stored_val(fname):
            ck = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu", weights_only=False)
            for key, state in ck.get("callbacks", {}).items():
                if "'monitor': 'val_loss'" in str(key) and isinstance(state, dict):
                    s = state.get("current_score")
                    if s is not None:
                        return float(s)
            return float("inf")
        chosen = min(ckpts, key=stored_val)
    else:
        raise ValueError(policy)

    model = CARL.load_from_checkpoint(os.path.join(ckpt_dir, chosen), map_location="cpu")
    model.eval()
    with open(os.path.join(carl_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, chosen


def carl_ratio(model, scaler, kinematics, batch=8192, clip=1e-7):
    """r(x) = s/(1-s) from a CARL classifier score s, on CPU, in batches."""
    X = scaler.transform(kinematics[FEATURES_4L].to_numpy())
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for a in range(0, len(X), batch):
            xb = torch.tensor(X[a:a + batch], dtype=torch.float32)
            s = model(xb).flatten().numpy().astype(np.float64)
            s = np.clip(s, clip, 1.0 - clip)
            out[a:a + batch] = s / (1.0 - s)
    return out


def nn_log_density_ratio(r_sig_sm, r_sbi_sm, xs, mu_grid):
    """Tae's mixture (his nsbi notebook, cell 9): the density ratio p(x|mu)/p_bkg(x)
    composed from the two SM-point network ratios and the component cross-sections."""
    mu = np.asarray(mu_grid, dtype=np.float64)
    m_sig = mu - np.sqrt(mu)
    m_sbi = np.sqrt(mu)
    m_bkg = 1.0 - np.sqrt(mu)
    num = (xs["sig"] * m_sig[None, :] * r_sig_sm[:, None]
           + xs["sbi"] * m_sbi[None, :] * r_sbi_sm[:, None]
           + xs["bkg"] * m_bkg[None, :])
    nu_mu = mu * xs["sig"] + np.sqrt(mu) * xs["int"] + xs["bkg"]
    ratio = num / nu_mu[None, :]
    return np.log(np.clip(ratio, _TINY, None)), nu_mu


# ---------------------------------------------------------------------------------------
# toys
# ---------------------------------------------------------------------------------------
def toys_legacy_block(w, n_toys, seed):
    """Tae's recipe, byte-compatible: one torch.manual_seed(seed), one block of draws.
    Returns a dense float64 array (n_toys, N)."""
    return np.asarray(toys.generate(w, n_toys=n_toys, random_state=seed), dtype=np.float64)


def toys_streams(w, n_toys, seed):
    """One independent numpy random stream per toy (SeedSequence.spawn).
    Returns a scipy.sparse CSR matrix (n_toys, N) of the per-event counts."""
    from scipy import sparse
    streams = np.random.SeedSequence(seed).spawn(n_toys)
    indptr = [0]
    indices, data = [], []
    for ss in streams:
        k = np.random.default_rng(ss).poisson(w)
        nz = np.nonzero(k)[0]
        indices.append(nz)
        data.append(k[nz].astype(np.float64))
        indptr.append(indptr[-1] + len(nz))
    return sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=(n_toys, len(w)),
    )


# ---------------------------------------------------------------------------------------
# scans
# ---------------------------------------------------------------------------------------
def hist_fill_unit(events, w_sm, w_scan):
    """m4l histogram templates at luminosity 1 fb^-1 (hist library, Tae's binning)."""
    import hist as hist_lib
    axis = hist_lib.axis.Regular(N_BINS, M4L_LO, M4L_HI, transform=hist_lib.axis.transform.log)
    m4l = events.kinematics["4l_mass"].values

    def fill(weights):
        h = hist_lib.Hist(axis)
        h.fill(m4l, weight=np.asarray(weights))
        return h.values()

    nu_asimov_unit = fill(w_sm)
    nu_exp_unit = np.stack([fill(w_scan[:, m]) for m in range(w_scan.shape[1])], 1)
    return nu_asimov_unit, nu_exp_unit, np.asarray(axis.edges), m4l


def hist_scan(events, w_sm, w_scan, L, n_toys, seed, legacy_seed_block):
    """Raw -2 ln L curves for the histogram analysis at luminosity L.
    Returns (nll_asimov (n_mu,), nll_toys (n_mu, n_toys)) as torch tensors."""
    nu_asimov_unit, nu_exp_unit, edges, m4l = hist_fill_unit(events, w_sm, w_scan)
    nu_asimov = nu_asimov_unit * L
    nu_exp = nu_exp_unit * L

    if legacy_seed_block:
        import hist as hist_lib
        k = toys_legacy_block(w_sm * L, n_toys, seed)  # (n_toys, N) dense
        axis = hist_lib.axis.Regular(N_BINS, M4L_LO, M4L_HI, transform=hist_lib.axis.transform.log)

        def fill(weights):
            h = hist_lib.Hist(axis)
            h.fill(m4l, weight=np.asarray(weights))
            return h.values()

        n_obs = np.stack([fill(k[i]) for i in range(n_toys)], 1)  # (n_bins, n_toys)
    else:
        K = toys_streams(w_sm * L, n_toys, seed)  # sparse (n_toys, N)
        # per-toy fill via bincount on precomputed bin indices (fast path); indices
        # verified against the hist-library fill in `main` before any sweep
        bin_idx = np.digitize(m4l, edges) - 1  # 0..n_bins-1 inside, -1/n_bins outside
        cols = []
        Kcsr = K.tocsr()
        for i in range(n_toys):
            sl = slice(Kcsr.indptr[i], Kcsr.indptr[i + 1])
            ev = Kcsr.indices[sl]
            cnt = Kcsr.data[sl]
            inside = (bin_idx[ev] >= 0) & (bin_idx[ev] < N_BINS)
            cols.append(np.bincount(bin_idx[ev][inside], weights=cnt[inside], minlength=N_BINS))
        n_obs = np.stack(cols, 1)

    nll_asimov = -2 * poisson_log_likelihood_scan(nu_asimov, nu_exp)
    nll_toys = -2 * poisson_log_likelihood_scan_toys(torch.tensor(n_obs), nu_exp)
    return nll_asimov, nll_toys


def unbinned_scan(w_sm, ln_mat, nu_mu_unit, L, n_toys, seed, legacy_seed_block,
                  shape_only, matrix_kind, mu_chunk=32):
    """Raw -2 ln L curves for the unbinned (per-event) likelihood at luminosity L.

    ln_mat : (N, n_mu) per-event matrix, of one of two kinds:
        matrix_kind='weight'  ln w_i(mu), the per-event expected weight at unit
                              luminosity ('nsbi-exact' full likelihood — self-consistent,
                              so the Asimov best fit is exactly mu = 1);
        matrix_kind='ratio'   ln of the normalised density ratio p(x|mu)/p_ref(x)
                              ('nsbi-nn', and any --shape-only run).
    nu_mu_unit : (n_mu,) expected total yield at unit luminosity.

    Full likelihood (every event its own Poisson bin; terms constant in mu kept):
        weight kind:  -2 ln L(mu) = 2*[ L*nu(mu) - sum_i k_i*ln(L*w_i(mu)) ]
        ratio  kind:  -2 ln L(mu) = 2*[ L*nu(mu) - N*ln(L*nu(mu)) - sum_i k_i*ln r_i(mu) ]
        (the two agree up to a mu-independent constant when the inputs describe the same
        model, since L*w_i(mu) = L*nu(mu) * p(x_i|mu)/nu-normalisation)
    Shape-only (Tae's committed statistic, ratio kind only):
        -2 ln L(mu) = -2*sum_i k_i*ln r(x_i|mu)
    """
    if shape_only and matrix_kind != "ratio":
        raise ValueError("--shape-only is defined on the density ratio")
    n_mu = ln_mat.shape[1]

    if legacy_seed_block:
        K = toys_legacy_block(w_sm * L, n_toys, seed)  # dense (n_toys, N)
        n_tot = K.sum(axis=1)
        contrib = np.empty((n_toys, n_mu))
        for a in range(0, n_mu, mu_chunk):
            contrib[:, a:a + mu_chunk] = K @ ln_mat[:, a:a + mu_chunk]
    else:
        K = toys_streams(w_sm * L, n_toys, seed)  # sparse (n_toys, N)
        n_tot = np.asarray(K.sum(axis=1)).ravel()
        contrib = np.asarray(K @ ln_mat)

    pos = w_sm > 0
    asimov_contrib = (w_sm[pos] * L) @ ln_mat[pos, :]
    n_tot_asimov = w_sm.sum() * L

    if shape_only:
        nll_asimov = -2.0 * asimov_contrib
        nll_toys = -2.0 * contrib
    elif matrix_kind == "weight":
        nll_asimov = 2.0 * (L * nu_mu_unit - asimov_contrib - n_tot_asimov * np.log(L))
        nll_toys = 2.0 * (L * nu_mu_unit[None, :] - contrib - n_tot[:, None] * np.log(L))
    else:  # matrix_kind == "ratio"
        ln_nuL = np.log(L * nu_mu_unit)
        nll_asimov = 2.0 * (L * nu_mu_unit - n_tot_asimov * ln_nuL - asimov_contrib)
        nll_toys = 2.0 * (L * nu_mu_unit[None, :] - n_tot[:, None] * ln_nuL[None, :] - contrib)

    return torch.tensor(nll_asimov), torch.tensor(nll_toys.T)  # (n_mu,), (n_mu, n_toys)


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------
def _save(out_dir, L, mu_grid, nll_asimov, nll_toys, meta):
    d = os.path.join(out_dir, f"lumi_{L:g}")
    os.makedirs(d, exist_ok=True)
    torch.save(nll_asimov, os.path.join(d, "nll_asimov.pt"))
    torch.save(nll_toys, os.path.join(d, "nll_toys.pt"))
    torch.save(torch.as_tensor(np.asarray(mu_grid)), os.path.join(d, "mu_scan.pt"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def _git_hash(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", choices=["hist", "nsbi-exact", "nsbi-nn"])
    p.add_argument("--lumis", type=float, nargs="+", default=[1., 10., 30., 100., 140., 200., 300., 500., 700., 1000., 3000.])
    p.add_argument("--n-toys", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mu-min", type=float, default=0.0)
    p.add_argument("--mu-max", type=float, default=4.0)
    p.add_argument("--mu-points", type=int, default=161)
    p.add_argument("--n-rows", type=float, default=3e5, help="events read from the SBI CSV")
    p.add_argument("--events-pkl", default=None,
                   help="use a pickled Process (e.g. Tae's events_numerator_test.pkl) instead of the CSV")
    p.add_argument("--shape-only", action="store_true",
                   help="Tae's committed statistic: no event-count term (diagnostic only)")
    p.add_argument("--shape-norm", choices=["xs", "sample"], default="xs",
                   help="nsbi-exact --shape-only: normalise the density with the"
                        " cross-section table (Tae's construction) or on the event sample")
    p.add_argument("--legacy-seed-block", action="store_true",
                   help="Tae's toy seeding (one torch.manual_seed block), for closure tests")
    p.add_argument("--checkpoint-policy", choices=["largest-epoch", "best-val"], default="best-val")
    p.add_argument("--checkpoint-dir", default=os.path.join(SUBREPO, "run", "h4l"))
    p.add_argument("--sig-checkpoint-dir", default=None,
                   help="override for the sig_over_bkg run dir (default <checkpoint-dir>/sig_over_bkg)")
    p.add_argument("--sbi-checkpoint-dir", default=None,
                   help="override for the sbi_over_bkg run dir (default <checkpoint-dir>/sbi_over_bkg)")
    p.add_argument("--sig-ratio", choices=["nn", "exact"], default="nn",
                   help="nsbi-nn only: take r_sig/bkg from the network or from the exact matrix"
                        " elements (isolates the sbi_over_bkg network's contribution)")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "zz4l", "ggzz4l_sm"))
    p.add_argument("--out", required=True)
    p.add_argument("--download-checkpoints", action="store_true")
    p.add_argument("--share-token", default="Shv8zta4oZ1aOL4")
    p.add_argument("--include-train-pkls", action="store_true")
    args = p.parse_args(argv)

    if args.download_checkpoints:
        download_share(args.share_token, args.out, args.include_train_pkls)
        return

    if args.analysis is None:
        p.error("--analysis is required (unless --download-checkpoints)")

    if args.legacy_seed_block:
        # Tae's notebooks built the grid as torch.linspace under a float32 default dtype;
        # reproduce those exact grid values so the closure comparison is bit-level
        mu_grid = torch.linspace(args.mu_min, args.mu_max, args.mu_points,
                                 dtype=torch.float32).numpy().astype(np.float64)
    else:
        mu_grid = np.linspace(args.mu_min, args.mu_max, args.mu_points)
    meta_common = dict(
        analysis=args.analysis, n_toys=args.n_toys, seed=args.seed,
        mu_grid=[args.mu_min, args.mu_max, args.mu_points],
        shape_only=args.shape_only, legacy_seed_block=args.legacy_seed_block,
        events_pkl=args.events_pkl, n_rows=None if args.events_pkl else int(args.n_rows),
        container_repo=_git_hash(ROOT), subrepo=_git_hash(SUBREPO),
        checkpoint_policy=args.checkpoint_policy if args.analysis == "nsbi-nn" else None,
    )

    xs = load_xs(args.data_dir)
    if args.events_pkl:
        with open(args.events_pkl, "rb") as f:
            events = pickle.load(f)
        print(f"events: {len(events.weights):,} from {args.events_pkl} "
              f"(sum of weights {events.weights.sum():.4f} fb)")
    else:
        events, _ = load_sbi_events(args.data_dir, args.n_rows)
        print(f"events: {len(events.weights):,} from the SBI CSV "
              f"(sum of weights {events.weights.sum():.4f} fb)")

    w_sm, w_scan = unit_weights(events, mu_grid)

    if args.analysis == "hist":
        if "4l_mass" not in events.kinematics:
            raise SystemExit("the hist analysis needs the 4l_mass column (CSV path, not pkl)")
        # cross-check the bincount fast path against the hist-library fill once
        import hist as hist_lib
        axis = hist_lib.axis.Regular(N_BINS, M4L_LO, M4L_HI, transform=hist_lib.axis.transform.log)
        m4l = events.kinematics["4l_mass"].values
        h = hist_lib.Hist(axis); h.fill(m4l, weight=w_sm)
        bin_idx = np.digitize(m4l, np.asarray(axis.edges)) - 1
        inside = (bin_idx >= 0) & (bin_idx < N_BINS)
        bc = np.bincount(bin_idx[inside], weights=w_sm[inside], minlength=N_BINS)
        assert np.allclose(h.values(), bc), "bincount fill disagrees with the hist-library fill"

        for L in args.lumis:
            nll_a, nll_t = hist_scan(events, w_sm, w_scan, L, args.n_toys, args.seed, args.legacy_seed_block)
            _, mu_hat = t_curves(nll_a.numpy(), mu_grid)
            print(f"hist  L={L:>6g} fb^-1  Asimov mu_hat = {mu_hat:.3f}")
            _save(args.out, L, mu_grid, nll_a, nll_t, dict(meta_common, lumi=L))

    else:
        if args.analysis == "nsbi-exact":
            if args.shape_only:
                nu_sample = w_scan.sum(axis=0) if args.shape_norm == "sample" else None
                ln_mat, nu_mu = exact_log_density_ratio(events, xs, mu_grid, nu_sample=nu_sample)
                matrix_kind = "ratio"
            else:
                ln_mat = np.log(np.clip(w_scan, _TINY, None))
                nu_mu = w_scan.sum(axis=0)  # sample-based yield: self-consistent, mu_hat(Asimov)=1 exactly
                matrix_kind = "weight"
        else:  # nsbi-nn
            sbi_dir = args.sbi_checkpoint_dir or os.path.join(args.checkpoint_dir, "sbi_over_bkg")
            model_sbi, scaler_sbi, ck_sbi = load_carl(sbi_dir, args.checkpoint_policy)
            r_sbi = carl_ratio(model_sbi, scaler_sbi, events.kinematics)
            meta_common["checkpoints"] = {"sbi_over_bkg": os.path.join(sbi_dir, ck_sbi)}
            if args.sig_ratio == "nn":
                sig_dir = args.sig_checkpoint_dir or os.path.join(args.checkpoint_dir, "sig_over_bkg")
                model_sig, scaler_sig, ck_sig = load_carl(sig_dir, args.checkpoint_policy)
                r_sig = carl_ratio(model_sig, scaler_sig, events.kinematics)
                meta_common["checkpoints"]["sig_over_bkg"] = os.path.join(sig_dir, ck_sig)
            else:
                # exact r_sig/bkg from the matrix elements; the cross-section normalisation
                # equals the density normalisation because the flat prefactors are common
                # to all components (verified: E_bkg[r_raw] matches xs_sbi/xs_bkg to 0.2%)
                comp = events.components
                msq_sig = comp[mcfm.csv_component_sm[mcfm.Component.SIG]].to_numpy(dtype=np.float64)
                msq_bkg = np.clip(comp[mcfm.csv_component_sm[mcfm.Component.BKG]].to_numpy(dtype=np.float64),
                                  _TINY, None)
                r_sig = np.clip((msq_sig / msq_bkg) * (xs["bkg"] / xs["sig"]), 0.0, 1e12)
                meta_common["checkpoints"]["sig_over_bkg"] = "exact matrix elements"
            print("ratios:", meta_common["checkpoints"])
            ln_mat, nu_mu = nn_log_density_ratio(r_sig, r_sbi, xs, mu_grid)
            matrix_kind = "ratio"

        for L in args.lumis:
            nll_a, nll_t = unbinned_scan(w_sm, ln_mat, nu_mu, L, args.n_toys, args.seed,
                                         args.legacy_seed_block, args.shape_only, matrix_kind)
            _, mu_hat = t_curves(nll_a.numpy(), mu_grid)
            print(f"{args.analysis}  L={L:>6g} fb^-1  Asimov mu_hat = {mu_hat:.3f}")
            _save(args.out, L, mu_grid, nll_a, nll_t, dict(meta_common, lumi=L))


if __name__ == "__main__":
    main()
