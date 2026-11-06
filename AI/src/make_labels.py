"""
Step 1 of the AI solution: instance-mask pseudo-labels with SAM 2.1.

SAM 2.1 in "segment everything" mode (a grid of point prompts) runs on
overlapping 640x640 tiles. No manual polygons are needed, and the labels are
independent of the Non-AI pipeline.

Per image
---------
1. Resize so the long side is at most WORK_SIZE (2048). Objects are 30-80 px.
2. Tile 640x640 with stride 512. Tiles that are almost pure background are
   skipped, for speed only.
3. SAM 2.1 on each tile. Raw candidates (global coordinates, cropped masks,
   scores) are cached in <out>/raw/, so step 4 can be re-tuned without
   re-running SAM.
4. Global clean-up on the whole image:
   a. drop specks, huge masks and bright masks (nut holes / background)
   b. global mask-NMS: overlapping duplicates from neighbouring tiles
   c. u = size of ONE object (left-most mode of the log-area histogram)
   d. GROUP masks, holding >= 2 object-sized masks -> dropped
   e. PARTS inside a larger mask (< 0.6u, e.g. a screw head) -> dropped
   f. PARTS touching a larger mask (< 0.5u, e.g. a head that SAM cut off
      the shank) -> merged into the neighbour they touch most
5. Save polygons to <out>/<image>.json, a preview overlay and sam_counts.csv.

Usage:  python src/make_labels.py --data <dataset_root> --out labels_sam --only ScrewAndBolt_20240713
"""
import argparse
import csv
import glob
import json
import os
import pickle
import time

import cv2
import numpy as np

WORK_SIZE = 2048
TILE, STRIDE = 640, 512
MIN_AREA = 60
MAX_FRAC = 0.15        # of a tile
BRIGHT_FRAC = 0.85     # mean gray > 0.85 x paper => hole / background
NMS_IOU = 0.5
PART_INSIDE = 0.6      # x u
PART_TOUCH = 0.5       # x u


def load(path):
    img = cv2.imread(path)
    s = min(1.0, WORK_SIZE / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def tile_origins(n, tile=TILE, stride=STRIDE):
    if n <= tile:
        return [0]
    return list(range(0, n - tile, stride)) + [n - tile]


# ---------------------------------------------------------------- SAM pass
def run_sam(model, img, device):
    """Return raw candidates: dicts with x0, y0 (global), m (cropped bool), score."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 1,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    H, W = gray.shape
    cands = []
    for ty in tile_origins(H):
        for tx in tile_origins(W):
            if fg[ty:ty + TILE, tx:tx + TILE].mean() < 0.005:
                continue
            t = img[ty:ty + TILE, tx:tx + TILE]
            r = model(t, verbose=False, device=device)[0]
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy().astype(bool)[:, :t.shape[0], :t.shape[1]]
            scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(masks))
            for m, s in zip(masks, scores):
                a = int(m.sum())
                if a < MIN_AREA or a > MAX_FRAC * m.size:
                    continue
                ys, xs = np.nonzero(m)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                cands.append({"x0": tx + x0, "y0": ty + y0, "m": m[y0:y1, x0:x1].copy(),
                              "score": float(s)})
    return cands


# ---------------------------------------------------------- mask helpers
def box(c):
    return c["x0"], c["y0"], c["x0"] + c["m"].shape[1], c["y0"] + c["m"].shape[0]


def inter(a, b):
    ax0, ay0, ax1, ay1 = box(a)
    bx0, by0, bx1, by1 = box(b)
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0
    ma = a["m"][y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    mb = b["m"][y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    return int(np.logical_and(ma, mb).sum())


def union(a, b):
    ax0, ay0, ax1, ay1 = box(a)
    bx0, by0, bx1, by1 = box(b)
    x0, y0, x1, y1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
    m = np.zeros((y1 - y0, x1 - x0), bool)
    m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] |= a["m"]
    m[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] |= b["m"]
    return {"x0": x0, "y0": y0, "m": m, "score": max(a["score"], b["score"])}


def dilate(c, r=3):
    m = cv2.dilate(np.pad(c["m"], r).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    return {"x0": c["x0"] - r, "y0": c["y0"] - r, "m": m, "score": c["score"]}


def area(c):
    return int(c["m"].sum())


def mode_area(areas):
    """Area of one object: left-most strong mode of the log-area histogram."""
    areas = np.asarray(areas, float)
    if len(areas) < 5:
        return float(np.median(areas))
    hist, edges = np.histogram(np.log(areas), bins=40)
    sm = np.convolve(hist, [1, 2, 3, 2, 1], mode="same").astype(float)
    thr, peak = 0.35 * sm.max(), int(np.argmax(sm))
    for i in range(len(sm)):
        l = sm[i - 1] if i else -1
        r = sm[i + 1] if i < len(sm) - 1 else -1
        if sm[i] >= thr and sm[i] >= l and sm[i] >= r:
            peak = i
            break
    p = np.exp(0.5 * (edges[peak] + edges[peak + 1]))
    sel = areas[(areas > 0.7 * p) & (areas < 1.4 * p)]
    return float(np.median(sel)) if len(sel) else float(p)


# ---------------------------------------------------------- clean-up pass
def clean(cands, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = float(np.percentile(gray, 90))

    # a. bright masks = holes / background
    def mean_gray(c):
        x0, y0, x1, y1 = box(c)
        return gray[y0:y1, x0:x1][c["m"]].mean()
    cands = [c for c in cands if mean_gray(c) <= BRIGHT_FRAC * paper]

    # b. global NMS by score
    kept = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        ac = area(c)
        dup = False
        for k in kept:
            i = inter(c, k)
            if i and i / (ac + area(k) - i) > NMS_IOU:
                dup = True
                break
        if not dup:
            kept.append(c)
    if not kept:
        return [], 0.0

    # c. size of one object
    u = mode_area([area(c) for c in kept])

    # d./e. groups and parts inside larger masks
    areas = [area(c) for c in kept]
    contains = [[] for _ in kept]
    for i, ci in enumerate(kept):
        for j, cj in enumerate(kept):
            if i != j and areas[j] < areas[i] and inter(ci, cj) > 0.8 * areas[j]:
                contains[i].append(j)
    drop = set()
    for i in range(len(kept)):
        if sum(1 for j in contains[i] if 0.6 * u <= areas[j] <= 1.6 * u) >= 2:
            drop.add(i)                                    # group mask
    for i in range(len(kept)):
        if i not in drop:
            drop.update(j for j in contains[i] if areas[j] < PART_INSIDE * u)
    objs = [c for k, c in enumerate(kept) if k not in drop]

    # f. small parts touching a larger object -> merge
    objs.sort(key=area)
    merged = True
    while merged:
        merged = False
        for i, c in enumerate(objs):
            if area(c) >= PART_TOUCH * u:
                continue
            d = dilate(c)
            best, best_contact = None, 0
            for j, o in enumerate(objs):
                if j != i and area(o) >= PART_TOUCH * u:
                    contact = inter(d, o)
                    if contact > best_contact:
                        best, best_contact = j, contact
            if best is not None:
                objs[best] = union(objs[best], c)
                objs.pop(i)
                merged = True
                break
    objs = [o for o in objs if area(o) >= 0.25 * u]       # leftover specks
    return objs, u


def to_polygon(c):
    cnts, _ = cv2.findContours(c["m"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return (max(cnts, key=cv2.contourArea).reshape(-1, 2) + [c["x0"], c["y0"]]).tolist()


def preview(img, polys):
    over = img.copy()
    rng = np.random.default_rng(0)
    for p in polys:
        cv2.fillPoly(over, [np.array(p, np.int32)], rng.integers(60, 255, 3).tolist())
    out = cv2.addWeighted(img, 0.45, over, 0.55, 0)
    for p in polys:
        cv2.polylines(out, [np.array(p, np.int32)], True, (0, 0, 255), 1)
    cv2.putText(out, f"SAM labels: {len(polys)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="labels_sam")
    ap.add_argument("--model", default="weights/sam2.1_b.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--only", nargs="*", help="substrings of image names to process")
    args = ap.parse_args()
    for sub in ("raw", "preview"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    model = None
    files = sorted(glob.glob(os.path.join(args.data, "*", "*.jpg")))
    if args.only:
        files = [f for f in files if any(s in f for s in args.only)]
    rows = []
    for f in files:
        name = f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"
        stem = name.replace("/", "__")[:-4]
        img = load(f)
        raw = os.path.join(args.out, "raw", stem + ".pkl")
        t0 = time.time()
        if os.path.exists(raw):                                   # cached SAM output
            cands = pickle.load(open(raw, "rb"))
        else:
            if model is None:
                from ultralytics import SAM
                model = SAM(args.model)
            cands = run_sam(model, img, args.device)
            pickle.dump(cands, open(raw, "wb"))
        objs, u = clean(cands, img)
        polys = [to_polygon(o) for o in objs]
        json.dump({"image": name, "width": img.shape[1], "height": img.shape[0], "unit_area": u,
                   "objects": [{"polygon": p} for p in polys]},
                  open(os.path.join(args.out, stem + ".json"), "w"))
        cv2.imwrite(os.path.join(args.out, "preview", stem + ".jpg"), preview(img, polys))
        rows.append({"image": name, "predicted": len(polys)})
        print(f"{name:45s} sam_count={len(polys):4d}  raw={len(cands):5d}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(args.out, "sam_counts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "predicted"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
