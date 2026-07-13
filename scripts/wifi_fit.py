#!/usr/bin/env python3
"""
wifi ensembles (Benevedes & Thaler, arXiv:2506.00113) on the off-shell gg->H*->ZZ->4l sig/bkg ratio.
Plan item 8e; study note docs/phase3-fixes.md.

Engine only: fits the ensemble weights, builds the sandwich covariance, runs the mixture-fraction
coverage experiment, and writes results to output/wifi/results/. All plotting/interpretation lives in
notebooks/wifi_ensembles.ipynb.

WHAT THIS ASKS (foundations sec. 17.3): wifi's uncertainties are derived asymptotically. Our failure
mode IS the breakdown of asymptotics. So: does the Gong-Samaniego inflation restore coverage at small
N, or not? We scan N_mix from the non-asymptotic regime up to the asymptotic one and measure coverage.

Equations follow the paper:
  (7)/(8)  symmetrized MLC loss   L(w) = <-u + e^-u - 1>_n + <u + e^u - 1>_d,  u = w_i f_i(x)
  (14-17)  score a_i = -f_i(1 + e^-u)  [numerator],  b_i = f_i(1 + e^u)  [denominator]
  (15)     V_ij = <f_i f_j e^-u>_n + <f_i f_j e^u>_d          (Hessian; loss is convex in w)
  (11)/(18) C = V^-1 U V^-1,  U = Cov_n[a]/N_n + Cov_d[b]/N_d  (sandwich)
  (27)     mixture likelihood  log L(x|k) = log(k r(x) + (1-k))
  (22-24)  sigma_GS^2 = sigma_MLE^2 (1 + sigma_MLE^2 A_i C_ij A_j),  A_i = d2 logL / dk dw_i
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from physics.simulation import mcfm
from physics.simulation.msq import Component
from models.carl import CARL

FEATURES = [f"l{i}_{v}" for i in range(1, 5) for v in ("pt", "eta", "phi", "energy")]
LOGIT_CLIP = 1e-6          # keep logit(s) finite
U_CLIP = 30.0              # keep exp(+-u) finite in the loss/score/Hessian


# ---------------------------------------------------------------- basis networks

def load_member(member_dir):
    """Load one basis network f_i and its scaler. Best-val_loss checkpoint."""
    ckpt_dir = None
    for root, _, files in os.walk(member_dir):
        if any(f.endswith(".ckpt") for f in files):
            ckpt_dir = root
            break
    if ckpt_dir is None:
        raise FileNotFoundError(f"no checkpoint under {member_dir}")

    # filenames are "epoch=NN-val_loss=X.XX.ckpt" (and one train_loss checkpoint) -> pick best val_loss
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and "val_loss" in f]
    if not ckpts:
        raise FileNotFoundError(f"no val_loss checkpoint under {ckpt_dir}")
    best = min(ckpts, key=lambda f: float(f.split("val_loss=")[1].removesuffix(".ckpt")))

    model = CARL.load_from_checkpoint(os.path.join(ckpt_dir, best), map_location="cpu")
    model.eval()

    with open(os.path.join(member_dir, "scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)

    return model, scaler, os.path.join(ckpt_dir, best)


def basis_logits(models_scalers, X):
    """f_i(x) = logit of member i's classifier output = its estimate of log r(x). Shape (N, M)."""
    out = np.empty((len(X), len(models_scalers)), dtype=np.float64)
    for i, (model, scaler) in enumerate(models_scalers):
        Xs = torch.tensor(scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            s = model(Xs).flatten().numpy().astype(np.float64)
        s = np.clip(s, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
        out[:, i] = np.log(s / (1.0 - s))
    return out


# ---------------------------------------------------------------- wifi weight fit

def sym_mlc_loss(w, f_n, f_d):
    u_n = np.clip(f_n @ w, -U_CLIP, U_CLIP)
    u_d = np.clip(f_d @ w, -U_CLIP, U_CLIP)
    return float(np.mean(-u_n + np.exp(-u_n) - 1.0) + np.mean(u_d + np.exp(u_d) - 1.0))


def sym_mlc_grad_hess(w, f_n, f_d):
    """Eqs. (14)-(17). Returns gradient, Hessian V, and the per-event scores a, b."""
    u_n = np.clip(f_n @ w, -U_CLIP, U_CLIP)
    u_d = np.clip(f_d @ w, -U_CLIP, U_CLIP)
    e_n = np.exp(-u_n)                                  # e^{-u} on numerator events
    e_d = np.exp(u_d)                                   # e^{+u} on denominator events

    a = -f_n * (1.0 + e_n)[:, None]                     # (N_n, M)   Eq. (16)
    b = f_d * (1.0 + e_d)[:, None]                      # (N_d, M)   Eq. (17)

    grad = a.mean(axis=0) + b.mean(axis=0)              # Eq. (14)
    V = (f_n * e_n[:, None]).T @ f_n / len(f_n) \
        + (f_d * e_d[:, None]).T @ f_d / len(f_d)       # Eq. (15)
    return grad, V, a, b


def fit_weights(f_n, f_d, max_iter=100, tol=1e-10, verbose=False):
    """Newton's method on the convex symmetrized MLC loss. Start from the naive ensemble w=1/M."""
    M = f_n.shape[1]
    w = np.full(M, 1.0 / M)

    for it in range(max_iter):
        grad, V, _, _ = sym_mlc_grad_hess(w, f_n, f_d)
        step = np.linalg.solve(V, grad)

        # backtracking line search: guarantees descent on a convex objective
        t, base = 1.0, sym_mlc_loss(w, f_n, f_d)
        while t > 1e-8 and sym_mlc_loss(w - t * step, f_n, f_d) > base:
            t *= 0.5
        w = w - t * step

        gnorm = float(np.max(np.abs(grad)))
        if verbose:
            print(f"  newton {it:2d}  loss={sym_mlc_loss(w, f_n, f_d):.8f}  |grad|={gnorm:.3e}  t={t:.3f}")
        if gnorm < tol:
            break

    return w


def weight_covariance(w, f_n, f_d):
    """Sandwich estimator, Eqs. (11) + (18)."""
    _, V, a, b = sym_mlc_grad_hess(w, f_n, f_d)
    U = np.cov(a, rowvar=False) / len(a) + np.cov(b, rowvar=False) / len(b)
    Vinv = np.linalg.inv(V)
    return Vinv @ U @ Vinv, V, U


# ---------------------------------------------------------------- mixture-fraction inference

def kappa_mle(r, lo=1e-6, hi=1.0 - 1e-6):
    """argmax_k sum log(k r + (1-k)), Eq. (27). Monotone score -> bisection; boundary-aware."""
    def score(k):
        return float(np.sum((r - 1.0) / (k * r + (1.0 - k))))

    s_lo, s_hi = score(lo), score(hi)
    if s_lo <= 0:                      # decreasing everywhere -> MLE at 0
        return 0.0, True
    if s_hi >= 0:                      # increasing everywhere -> MLE at 1
        return 1.0, True

    for _ in range(200):               # score is strictly decreasing in k
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def kappa_uncertainties(k_hat, r, f, C):
    """sigma_MLE (Eq. 23), A_i (Eq. 24), sigma_GS (Eq. 22).

    With rho = r(x) and g = k rho + (1-k):
        d2 logL / dk2      = -sum (rho-1)^2 / g^2
        d2 logL / dk dw_i  =  sum rho f_i / g^2      (using drho/dw_i = rho f_i)
    """
    g = k_hat * r + (1.0 - k_hat)
    g = np.maximum(g, 1e-12)

    info = float(np.sum((r - 1.0) ** 2 / g ** 2))          # = -d2logL/dk2
    if info <= 0:
        return np.nan, np.nan, np.nan
    var_mle = 1.0 / info

    A = (r / g ** 2) @ f                                    # (M,)  Eq. (24)
    inflation = float(A @ C @ A)
    var_gs = var_mle * (1.0 + var_mle * inflation)          # Eq. (22)
    return var_mle, var_gs, inflation


# ---------------------------------------------------------------- experiment

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="output/wifi/h4l")
    ap.add_argument("--out-dir", default="output/wifi/results")
    ap.add_argument("--data-dir", default="data/zz4l/ggzz4l_sm")
    ap.add_argument("--members", type=int, nargs="+", default=[4, 8, 16],
                    help="ensemble sizes M to evaluate (subsets of the trained members)")
    ap.add_argument("--n-fit", type=int, default=100_000, help="i.i.d. events per class in D_fit")
    ap.add_argument("--n-pool", type=int, default=200_000, help="i.i.d. events per class in the D_exp pool")
    ap.add_argument("--n-mix", type=int, nargs="+", default=[30, 100, 300, 1000, 3000, 10000],
                    help="experiment sizes N_mix (the non-asymptotic -> asymptotic axis)")
    ap.add_argument("--kappa-true", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260713)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- basis networks
    member_dirs = sorted(
        os.path.join(args.run_dir, d) for d in os.listdir(args.run_dir) if d.startswith("member_")
    )
    M_max = max(args.members)
    if len(member_dirs) < M_max:
        raise SystemExit(f"need >= {M_max} members, found {len(member_dirs)} in {args.run_dir}")
    member_dirs = member_dirs[:M_max]

    models_scalers, ckpts = [], []
    for d in member_dirs:
        model, scaler, ckpt = load_member(d)
        models_scalers.append((model, scaler))
        ckpts.append(ckpt)
        print(f"loaded {os.path.basename(d)}: {os.path.basename(ckpt)}")

    # ---- data: the TEST split only (disjoint from what trained f_i and from validation)
    print("loading test splits ...")
    # 4l_mass is carried along for the diagnostic plots only - it is NOT a training feature
    sig = mcfm.from_csv(file_path=os.path.join(args.data_dir, "ggzz4l_sig.csv"), kinematics=FEATURES + ["4l_mass"])
    bkg = mcfm.from_csv(file_path=os.path.join(args.data_dir, "ggzz4l_bkg.csv"), kinematics=FEATURES + ["4l_mass"])
    _, _, sig_test = sig.split(train_size=6, val_size=2, test_size=2)
    _, _, bkg_test = bkg.split(train_size=6, val_size=2, test_size=2)
    print(f"  test events: sig {len(sig_test.weights)}  bkg {len(bkg_test.weights)}")

    # split the test sample in half: D_fit (fits w) and the D_exp pool (the experiments). Disjoint.
    def halves(proc):
        n = len(proc.weights)
        return proc[: n // 2], proc[n // 2:]

    sig_fit_src, sig_pool_src = halves(sig_test)
    bkg_fit_src, bkg_pool_src = halves(bkg_test)

    # i.i.d. draws: the paper's derivation assumes unweighted samples (Process.unweight already does
    # exactly this - draw with replacement in proportion to weight).
    sig_fit = sig_fit_src.unweight(args.n_fit, random_state=args.seed + 1)
    bkg_fit = bkg_fit_src.unweight(args.n_fit, random_state=args.seed + 2)
    sig_pool = sig_pool_src.unweight(args.n_pool, random_state=args.seed + 3)
    bkg_pool = bkg_pool_src.unweight(args.n_pool, random_state=args.seed + 4)

    # ---- truth ratio (the control the paper could not have): r_true = (msq_sig/msq_bkg) / norm,
    # normalised so that E_bkg[r_true] = 1, which is exactly the paper's Eq. (6) constraint.
    def msq_ratio(proc):
        num = proc.components[mcfm.csv_component_sm[Component.SIG]].to_numpy(dtype=np.float64)
        den = proc.components[mcfm.csv_component_sm[Component.BKG]].to_numpy(dtype=np.float64)
        return num / np.maximum(den, 1e-300)

    norm = float(np.mean(msq_ratio(bkg_pool)))     # E_bkg[msq_sig/msq_bkg]
    r_true_sig_pool = msq_ratio(sig_pool) / norm
    r_true_bkg_pool = msq_ratio(bkg_pool) / norm
    print(f"truth-ratio normalisation E_bkg[msq_sig/msq_bkg] = {norm:.6e}")
    print(f"  closure  E_bkg[r_true] = {np.mean(r_true_bkg_pool):.6f} (must be 1)")

    # ---- basis evaluation (once; everything downstream just indexes these)
    print("evaluating basis networks ...")
    f_fit_n = basis_logits(models_scalers, sig_fit.kinematics[FEATURES].to_numpy())
    f_fit_d = basis_logits(models_scalers, bkg_fit.kinematics[FEATURES].to_numpy())
    f_pool_n = basis_logits(models_scalers, sig_pool.kinematics[FEATURES].to_numpy())
    f_pool_d = basis_logits(models_scalers, bkg_pool.kinematics[FEATURES].to_numpy())

    results = {"config": vars(args), "checkpoints": ckpts,
               "truth_norm": norm, "closure_E_bkg_r_true": float(np.mean(r_true_bkg_pool)),
               "ensembles": {}, "coverage": []}

    for M in args.members:
        print(f"\n=== M = {M} ===")
        fn, fd = f_fit_n[:, :M], f_fit_d[:, :M]

        w_hat = fit_weights(fn, fd, verbose=True)
        C, V, U = weight_covariance(w_hat, fn, fd)
        w_naive = np.full(M, 1.0 / M)

        print(f"  w_hat      = {np.array2string(w_hat, precision=3)}  (sum {w_hat.sum():.3f})")
        print(f"  sigma(w)   = {np.array2string(np.sqrt(np.diag(C)), precision=3)}")
        print(f"  loss: naive {sym_mlc_loss(w_naive, fn, fd):.6f} -> wifi {sym_mlc_loss(w_hat, fn, fd):.6f}")

        results["ensembles"][str(M)] = {
            "w_hat": w_hat.tolist(), "C": C.tolist(),
            "sigma_w": np.sqrt(np.diag(C)).tolist(),
            "loss_naive": sym_mlc_loss(w_naive, fn, fd),
            "loss_wifi": sym_mlc_loss(w_hat, fn, fd),
        }

        # log-ratio estimates on the experiment pool, for each estimator of r
        r_pool = {
            "wifi":  (np.exp(np.clip(f_pool_n[:, :M] @ w_hat, -U_CLIP, U_CLIP)),
                      np.exp(np.clip(f_pool_d[:, :M] @ w_hat, -U_CLIP, U_CLIP))),
            "naive": (np.exp(np.clip(f_pool_n[:, :M] @ w_naive, -U_CLIP, U_CLIP)),
                      np.exp(np.clip(f_pool_d[:, :M] @ w_naive, -U_CLIP, U_CLIP))),
            "truth": (r_true_sig_pool, r_true_bkg_pool),
        }

        for k_true in args.kappa_true:
            for n_mix in args.n_mix:
                acc = {m: {"cov_mle": 0, "cov_gs": 0, "khat": [], "smle": [], "sgs": [], "bnd": 0}
                       for m in r_pool}

                for _ in range(args.n_trials):
                    n_sig = int(rng.binomial(n_mix, k_true))
                    idx_s = rng.integers(0, args.n_pool, size=n_sig)
                    idx_b = rng.integers(0, args.n_pool, size=n_mix - n_sig)

                    for meth, (rs, rb) in r_pool.items():
                        r = np.concatenate([rs[idx_s], rb[idx_b]])
                        k_hat, at_bnd = kappa_mle(r)

                        f_trial = np.concatenate([f_pool_n[idx_s, :M], f_pool_d[idx_b, :M]])
                        v_mle, v_gs, _ = kappa_uncertainties(k_hat, r, f_trial, C)
                        if not np.isfinite(v_mle):
                            continue

                        # The Gong-Samaniego inflation is only DEFINED for the fitted-weight ratio:
                        # C is the covariance of THOSE weights. A naive 1/M ensemble has no fitted
                        # weights and hence no such covariance - a practitioner using it would quote
                        # sigma_MLE, which is exactly the overconfidence the paper reports. The truth
                        # ratio has no estimation uncertainty at all. So GS is wifi-only, by construction.
                        s_mle = np.sqrt(v_mle)
                        s_gs = np.sqrt(v_gs) if meth == "wifi" else np.nan
                        acc[meth]["khat"].append(k_hat)
                        acc[meth]["smle"].append(s_mle)
                        acc[meth]["sgs"].append(s_gs)
                        acc[meth]["bnd"] += int(at_bnd)
                        # 68% Wald intervals, as in the paper
                        acc[meth]["cov_mle"] += int(abs(k_hat - k_true) <= s_mle)
                        if meth == "wifi":
                            acc[meth]["cov_gs"] += int(abs(k_hat - k_true) <= s_gs)

                for meth, a in acc.items():
                    n = len(a["khat"])
                    if n == 0:
                        continue
                    is_wifi = (meth == "wifi")
                    row = {
                        "M": M, "kappa_true": k_true, "n_mix": n_mix, "method": meth,
                        "n_trials": n,
                        "coverage_mle": a["cov_mle"] / n,
                        "coverage_gs": (a["cov_gs"] / n) if is_wifi else np.nan,
                        "coverage_err": float(np.sqrt(a["cov_mle"] / n * (1 - a["cov_mle"] / n) / n)),
                        "khat_mean": float(np.mean(a["khat"])),
                        "khat_std": float(np.std(a["khat"])),
                        "sigma_mle_mean": float(np.mean(a["smle"])),
                        "sigma_gs_mean": float(np.mean(a["sgs"])) if is_wifi else np.nan,
                        "inflation_mean": float(np.mean(np.array(a["sgs"]) / np.array(a["smle"]))) if is_wifi else np.nan,
                        "frac_at_boundary": a["bnd"] / n,
                    }
                    results["coverage"].append(row)

                pr = {m: (acc[m]["cov_mle"] / max(len(acc[m]["khat"]), 1),
                          acc[m]["cov_gs"] / max(len(acc[m]["khat"]), 1)) for m in acc}  # gs meaningful for wifi only
                print(f"  k={k_true:<5} N={n_mix:<6} "
                      f"truth {pr['truth'][0]:.3f} | naive {pr['naive'][0]:.3f} | "
                      f"wifi(MLE) {pr['wifi'][0]:.3f} -> wifi(GS) {pr['wifi'][1]:.3f}")

    # ---- diagnostics for the notebook: the estimated ratios against the TRUTH ratio, on a
    # subsample of the pool. This is the control the paper could not build (no truth ratio for jets);
    # it separates ratio NOISE (what wifi propagates) from ratio BIAS (what it explicitly does not).
    M_ref = max(args.members)
    w_ref = np.array(results["ensembles"][str(M_ref)]["w_hat"])
    n_diag = min(20_000, args.n_pool)
    sl = slice(0, n_diag)
    np.savez_compressed(
        os.path.join(args.out_dir, "diagnostics.npz"),
        M_ref=M_ref,
        f_pool_n=f_pool_n[sl, :M_ref], f_pool_d=f_pool_d[sl, :M_ref],
        r_true_n=r_true_sig_pool[sl], r_true_d=r_true_bkg_pool[sl],
        logr_wifi_n=f_pool_n[sl, :M_ref] @ w_ref, logr_wifi_d=f_pool_d[sl, :M_ref] @ w_ref,
        logr_naive_n=f_pool_n[sl, :M_ref].mean(axis=1), logr_naive_d=f_pool_d[sl, :M_ref].mean(axis=1),
        m4l_n=sig_pool.kinematics["4l_mass"].to_numpy()[sl],
        m4l_d=bkg_pool.kinematics["4l_mass"].to_numpy()[sl],
        C_ref=np.array(results["ensembles"][str(M_ref)]["C"]),
        w_ref=w_ref,
    )

    out_json = os.path.join(args.out_dir, "wifi_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    pd.DataFrame(results["coverage"]).to_csv(os.path.join(args.out_dir, "coverage.csv"), index=False)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
