"""
One filtered pseudo-labelling round on the TRAINING tiles.

A trained YOLO-seg model predicts on every training tile, and its predictions
are used to improve the existing labels. Raw predictions are never accepted as
they are, because a model would then reinforce its own mistakes.

Rules, applied per tile:
  KEEP     every existing label stays. Nothing is ever deleted.
  REFINE   a prediction replaces a label's outline only if conf >= 0.70,
           IoU >= 0.60 and its area is 1.10-1.40x the label's area. This
           repairs labels that are too small (e.g. a partly masked shank)
           and leaves good SAM outlines alone.
  MERGE    an unmatched prediction (conf >= 0.70) that touches a label and is
           smaller than 0.6x the median label area is a missing part, such as
           a shank beside a labelled head. It is joined to that label.
  ADD      an unmatched prediction becomes a new label only if conf >= 0.70,
           it touches no label, it does not touch the tile border, and it is
           at least 0.25x the median label area.
  SKIP     anything else, including large predictions touching a label,
           because those cases are ambiguous.

Only images/train is changed. Validation tiles and test images are not read,
so the held-out test stays independent of this round.

Usage:
    python src/pseudo_label_round.py --weights best.pt --dataset datasets/challenge_seg \
        --out datasets/challenge_seg_r2 --device 0
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

REFINE_CONF, REFINE_IOU, REFINE_RATIO = 0.70, 0.60, (1.10, 1.40)
ADD_CONF, ADD_MIN_AREA, MERGE_MAX_AREA, BORDER, TOUCH_PX = 0.70, 0.25, 0.60, 3, 5


def read_labels(path, w, h):
    polys = []
    if os.path.exists(path):
        for line in open(path):
            v = line.split()[1:]
            if len(v) >= 6:
                polys.append((np.array(v, float).reshape(-1, 2) * [w, h]).astype(np.float32))
    return polys


def raster(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m.astype(bool)


def to_line(poly, w, h):
    p = np.clip(poly / [w, h], 0, 1)
    return "0 " + " ".join(f"{v:.5f}" for v in p.reshape(-1))


def merge_tile(labels, preds, confs, shape, min_area):
    """Return (polygons, n_refined, n_added, n_merged) for one tile."""
    h, w = shape
    lab_m = [raster(p, shape) for p in labels]
    prd_m = [raster(p, shape) for p in preds]
    out = list(labels)
    pairs = []
    for j, pm in enumerate(prd_m):
        for i, lm in enumerate(lab_m):
            inter = np.logical_and(pm, lm).sum()
            if inter:
                pairs.append((inter / np.logical_or(pm, lm).sum(), i, j))
    used_l, used_p, refined = set(), set(), 0
    for iou, i, j in sorted(pairs, reverse=True):        # one-to-one matching
        if iou < REFINE_IOU or i in used_l or j in used_p:
            continue
        used_l.add(i)
        used_p.add(j)
        ratio = prd_m[j].sum() / max(1, lab_m[i].sum())
        if confs[j] >= REFINE_CONF and REFINE_RATIO[0] <= ratio <= REFINE_RATIO[1]:
            out[i] = preds[j]
            refined += 1
    kernel = np.ones((2 * TOUCH_PX + 1, 2 * TOUCH_PX + 1), np.uint8)
    added = merged = 0
    for j, pm in enumerate(prd_m):
        a = pm.sum()
        if j in used_p or confs[j] < ADD_CONF or a < ADD_MIN_AREA * min_area:
            continue
        near = cv2.dilate(pm.astype(np.uint8), kernel) > 0
        contact = [np.logical_and(near, raster(p, shape)).sum() for p in out]
        if contact and max(contact) > 0:
            if a < MERGE_MAX_AREA * min_area:
                i = int(np.argmax(contact))
                joined = cv2.dilate((raster(out[i], shape) | pm).astype(np.uint8), kernel)
                joined = cv2.erode(joined, kernel)
                cnts, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                out[i] = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                merged += 1
            continue
        x, y = preds[j][:, 0], preds[j][:, 1]
        if x.min() < BORDER or y.min() < BORDER or x.max() > w - BORDER or y.max() > h - BORDER:
            continue
        out.append(preds[j])
        added += 1
    return out, refined, added, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, help="process only N tiles (for testing)")
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(args.dataset, args.out)

    tiles = sorted(glob.glob(os.path.join(args.dataset, "images", "train", "*.jpg")))
    if args.limit:
        tiles = tiles[::max(1, len(tiles) // args.limit)][:args.limit]
    areas = []
    for t in tiles:
        h, w = cv2.imread(t).shape[:2]
        lab = t.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
        areas += [cv2.contourArea(p) for p in read_labels(lab, w, h)]
    min_area = float(np.median(areas)) if areas else 0.0

    model = YOLO(args.weights)
    totals = {"tiles": 0, "labels_before": 0, "labels_after": 0, "refined": 0, "added": 0, "merged": 0,
              "added_by_tile": {}}
    for t in tiles:
        img = cv2.imread(t)
        h, w = img.shape[:2]
        stem = os.path.basename(t)[:-4]
        labels = read_labels(os.path.join(args.dataset, "labels", "train", stem + ".txt"), w, h)
        r = model.predict(img, imgsz=640, conf=0.25, max_det=1000, retina_masks=True,
                          verbose=False, device=args.device)[0]
        preds, confs = [], []
        if r.masks is not None:
            for p, c in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
                if len(p) >= 3:
                    preds.append(np.asarray(p, np.float32))
                    confs.append(float(c))
        result, refined, added, joined = merge_tile(labels, preds, confs, (h, w), min_area)
        with open(os.path.join(args.out, "labels", "train", stem + ".txt"), "w") as fh:
            fh.write("\n".join(to_line(p, w, h) for p in result) + ("\n" if result else ""))
        totals["tiles"] += 1
        totals["labels_before"] += len(labels)
        totals["labels_after"] += len(result)
        totals["refined"] += refined
        totals["added"] += added
        totals["merged"] += joined
        if added:
            totals["added_by_tile"][stem] = added
    json.dump(totals, open(os.path.join(args.out, "pseudo_label_report.json"), "w"), indent=1)
    print({k: v for k, v in totals.items() if k != "added_by_tile"})


if __name__ == "__main__":
    main()
