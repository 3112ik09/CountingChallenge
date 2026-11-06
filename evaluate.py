"""
Compare one or more prediction CSVs against a reference CSV.

    python evaluate.py --gt reference_counts.csv \
        Non_AI/v1/results/predictions.csv Non_AI/v2/results/predictions.csv

reference_counts.csv columns: image,gt
predictions.csv columns:  image,predicted[,...]
The method name is given as NAME=path, or defaults to the folder two levels
above each predictions.csv (e.g. "v1", "v2").

Metrics: per-image accuracy = 1 - |pred - gt| / gt, mean / worst accuracy,
MAE, RMSE, signed bias (mean pred - gt), and #images below 95 %.
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", help="optional path to write the merged table (csv)")
    ap.add_argument("--split", help="optional split.json: adds a train/val/test column")
    ap.add_argument("preds", nargs="+")
    args = ap.parse_args()

    df = pd.read_csv(args.gt)[["image", "gt"]]
    names = []
    for p in args.preds:
        if "=" in p:                       # NAME=path/to/predictions.csv
            name, p = p.split("=", 1)
        else:
            name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))
        d = pd.read_csv(p)[["image", "predicted"]].rename(columns={"predicted": name})
        df = df.merge(d, on="image", how="left")
        names.append(name)

    if args.split:
        import json
        sp = json.load(open(args.split))
        df.insert(1, "split", df["image"].map({im: k for k, v in sp.items() for im in v}))
    for n in names:
        df[f"{n}_acc%"] = (1 - (df[n] - df["gt"]).abs() / df["gt"]) * 100

    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    print()
    print(f"{'method':8s} {'mean acc':>9s} {'worst':>7s} {'MAE':>6s} {'RMSE':>6s} {'bias':>6s} {'<95%':>5s}")
    for n in names:
        err = df[n] - df["gt"]
        acc = df[f"{n}_acc%"]
        print(f"{n:8s} {acc.mean():8.2f}% {acc.min():6.1f}% {err.abs().mean():6.2f} "
              f"{np.sqrt((err ** 2).mean()):6.2f} {err.mean():+6.2f} {(acc < 95).sum():5d}")
    if args.split:
        print("\nheld-out TEST images only:")
        t = df[df["split"] == "test"]
        for n in names:
            err = t[n] - t["gt"]
            print(f"{n:8s} {t[f'{n}_acc%'].mean():8.2f}% {t[f'{n}_acc%'].min():6.1f}% {err.abs().mean():6.2f}")
    if args.out:
        df.round(2).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
