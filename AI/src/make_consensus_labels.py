"""
Step 1b of the AI solution: consensus labels = SAM 2.1 masks + Non-AI v2 objects.

Why: SAM 2.1 "segment everything" alone was not reliable on two scene types.
  - pan-head screws: SAM masks the threaded shank and the head separately
    (or misses the dark head), so one screw became 2 labels
  - mixed scenes: SAM missed many black screws and split others
A model trained on those labels copied the errors (AI v1).

How: two independent sources are reconciled per object.
  SAM masks       - accurate boundaries, but sometimes fragments / misses
  Non-AI v2       - reliable object count, approximate boundaries
For each image (at the 2048-px label scale):
  1. Rasterise the SAM polygons; run Non-AI v2 to get its instance label image.
  2. Assign every SAM mask to the v2 object it overlaps most (>= 30 % of the
     SAM mask). SAM masks with no v2 object are "SAM-only" objects.
  3. Per v2 object:
       - SAM pieces assigned to it are UNIONED  -> fixes head/shank splits
       - if SAM covers < 60 % of the v2 object, the v2 mask is added too
         (e.g. the dark pan-head that SAM missed)
       - no SAM piece at all -> the v2 mask is used  -> fixes missed screws
  4. SAM-only objects are kept if they are at least 0.5x a typical object
     (objects the classical pipeline missed).
  5. Every disagreement is logged in consensus_report.csv for review.

The resulting labels depend partly on the Non-AI pipeline. This is disclosed
in the README.

Usage:
    python src/make_consensus_labels.py --data <dataset_root> --sam labels_sam --out labels_consensus
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "Non_AI", "v2"))
import count_objects as nonai  # noqa: E402

ASSIGN_MIN = 0.30     # SAM mask must overlap a v2 object by >= 30 % of its area
COVER_MIN = 0.60      # SAM pieces must cover >= 60 % of the v2 object, else add v2 mask
EXTRA_MIN = 0.50      # SAM-only object kept if >= 0.5 x median object area


def polygon_mask(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


def to_polygon(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c.reshape(-1, 2).tolist() if len(c) >= 3 else None


def consensus(img_path, sam_json):
    lab = json.load(open(sam_json))
    W, H = lab["width"], lab["height"]
    # Non-AI v2 at its own working size, label image rescaled to the label scale
    small = nonai.load(img_path)
    v2_count, v2_seg, _, _ = nonai.count_image(small)
    v2 = cv2.resize(v2_seg.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    v2_ids = [i for i in np.unique(v2) if i > 0]

    sam_masks = [polygon_mask(o["polygon"], (H, W)) for o in lab["objects"]]
    assigned = {i: [] for i in v2_ids}
    sam_only = []
    for k, m in enumerate(sam_masks):
        a = m.sum()
        if a == 0:
            continue
        ids, cnt = np.unique(v2[m], return_counts=True)
        sel = ids > 0
        ids, cnt = ids[sel], cnt[sel]
        if len(ids) and cnt.max() >= ASSIGN_MIN * a:
            assigned[ids[np.argmax(cnt)]].append(k)
        else:
            sam_only.append(k)

    objects, stats = [], {"merged_fragments": 0, "v2_added_missing": 0, "v2_extended": 0, "sam_only_kept": 0}
    areas = []
    for i in v2_ids:
        vm = v2 == i
        pieces = assigned[i]
        if not pieces:
            m = vm
            stats["v2_added_missing"] += 1
        else:
            m = np.zeros((H, W), bool)
            for k in pieces:
                m |= sam_masks[k]
            if len(pieces) > 1:
                stats["merged_fragments"] += len(pieces) - 1
            if (m & vm).sum() < COVER_MIN * vm.sum():
                m |= vm
                stats["v2_extended"] += 1
        areas.append(m.sum())
        objects.append(m)
    med = float(np.median(areas)) if areas else 0.0
    for k in sam_only:
        if sam_masks[k].sum() >= EXTRA_MIN * med:
            objects.append(sam_masks[k])
            stats["sam_only_kept"] += 1

    polys = [p for p in (to_polygon(m) for m in objects) if p is not None]
    out = {"image": lab["image"], "width": W, "height": H,
           "objects": [{"polygon": p} for p in polys]}
    stats.update({"sam_count": len(lab["objects"]), "v2_count": v2_count, "final_count": len(polys)})
    return out, stats


def preview(img, polys, title):
    over = img.copy()
    rng = np.random.default_rng(0)
    for p in polys:
        cv2.fillPoly(over, [np.array(p, np.int32)], rng.integers(60, 255, 3).tolist())
    out = cv2.addWeighted(img, 0.45, over, 0.55, 0)
    for p in polys:
        cv2.polylines(out, [np.array(p, np.int32)], True, (0, 0, 255), 1)
    cv2.putText(out, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sam", default="labels_sam")
    ap.add_argument("--out", default="labels_consensus")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)
    rows = []
    for sj in sorted(glob.glob(os.path.join(args.sam, "*.json"))):
        name = json.load(open(sj))["image"]
        img_path = os.path.join(args.data, name)
        out, st = consensus(img_path, sj)
        stem = os.path.basename(sj)[:-5]
        json.dump(out, open(os.path.join(args.out, stem + ".json"), "w"))
        big = cv2.imread(img_path)
        s = out["width"] / big.shape[1]
        img = cv2.resize(big, (out["width"], out["height"]), interpolation=cv2.INTER_AREA) if s < 1 else big
        cv2.imwrite(os.path.join(args.out, "preview", stem + ".jpg"),
                    preview(img, [o["polygon"] for o in out["objects"]],
                            f"consensus {st['final_count']} (SAM {st['sam_count']}, v2 {st['v2_count']})"))
        rows.append({"image": name, **st})
        print(f"{name:45s} SAM={st['sam_count']:4d} v2={st['v2_count']:4d} final={st['final_count']:4d} "
              f"merged={st['merged_fragments']:3d} added={st['v2_added_missing']:3d} "
              f"extended={st['v2_extended']:3d} sam_only={st['sam_only_kept']:3d}", flush=True)
    keys = ["image", "sam_count", "v2_count", "final_count", "merged_fragments",
            "v2_added_missing", "v2_extended", "sam_only_kept"]
    with open(os.path.join(args.out, "consensus_report.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
