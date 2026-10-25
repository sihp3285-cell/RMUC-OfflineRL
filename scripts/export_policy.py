"""Bundle a trained policy into a single deployable zip.

Collects exactly what deploy.py needs: the checkpoint + observation stats +
meta (and rtg_targets.txt for a Decision Transformer).

Usage:
    python scripts/export_policy.py rm_runs/sentry_iql
    python scripts/export_policy.py rm_runs/sentry_iql --ckpt ckpt_50000.pt -o sentry_policy.zip
"""
import argparse
import os
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="auto",
                    help="'auto' (default) picks best.pt, else final.pt. "
                         "These runs overfit within a few thousand steps, so "
                         "final.pt is reliably the WORST model in the directory "
                         "— never ship it unless you know why.")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    rd = args.run_dir
    if args.ckpt == "auto":
        args.ckpt = ("best.pt" if os.path.exists(os.path.join(rd, "best.pt"))
                     else "final.pt")
        print(f"[export] selected {args.ckpt}")
    want = [args.ckpt, "norm_stats.npz", "meta.json", "rtg_targets.txt"]
    files = [f for f in want if os.path.exists(os.path.join(rd, f))]
    required = {args.ckpt, "norm_stats.npz"}
    missing = required - set(files)
    if missing:
        raise SystemExit(f"missing required files in {rd}: {sorted(missing)}")

    out = args.out or os.path.join(
        os.path.dirname(rd.rstrip("/\\")) or ".",
        os.path.basename(rd.rstrip("/\\")) + "_policy.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            # store the chosen ckpt as best.pt: that is the name
            # deploy.load_policy looks for first
            arc = "best.pt" if f == args.ckpt else f
            z.write(os.path.join(rd, f), arc)
    size = os.path.getsize(out) / 1024
    print(f"exported {files} -> {out} ({size:.1f} KB)")
    print("Deploy: unzip, then MLPPolicyRunner(<folder>) — see README section 6.4.")


if __name__ == "__main__":
    main()
