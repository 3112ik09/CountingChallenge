"""
Step 2 of the AI solution: build a YOLO-seg dataset from the SAM 2.1 labels.

Split (by image, before tiling, so no tile of a test image is seen in training):
  test : one image per scene group. NEVER used for training or model selection.
  train: all other images, so every scene type is represented.
  val  : a random 10 % of the TRAIN tiles, used only to pick the best epoch.
         (AI v1 used whole val images, which removed the only pan-head
          training image; that is why v1 failed on pan-head screws.)

Training data = 640x640 tiles (stride 320) cut from the 2048-px images,
because objects are small (30-80 px) and dense. Instances cut by a tile edge
are kept if at least 25 % of the object is visible in the tile.

Output (zip it and upload to Colab):
  datasets/challenge_seg/
      images/{train,val}/*.jpg   labels/{train,val}/*.txt   (YOLO-seg tiles)
      test_full/*.jpg            full 2048-px test images, for tiled inference
      test_full_labels.csv       label count per test image
      data.yaml  split.json

Usage:  python src/build_dataset.py --data <dataset_root> --labels labels_consensus \
            --out datasets/challenge_seg --folder ScrewAndBolt_20240713
"""
import argparse
import csv
import glob
import json
import os
import shutil

import cv2
import numpy as np

TILE, STRIDE = 640, 320
MIN_VISIBLE = 0.25

# one image per scene group
TEST = ["Screws_2024_07_15/img5.jpg",                 # brass screws, white
        "ScrewAndBolt_20240713/20240713_193135.jpg",  # sparse pan-head screws
        "ScrewAndBolt_20240713/20240713_193659.jpg",  # sparse countersunk screws
        "ScrewAndBolt_20240713/20240713_193907.jpg",  # dense screw pile
        "ScrewAndBolt_20240713/20240713_194316.jpg",  # mixed screws + round heads
        "ScrewAndBolt_20240713/20240713_194630.jpg"]  # dense nut pile
VAL_TILE_FRAC = 0.10


def load(path, work=2048):
    img = cv2.imread(path)
    s = min(1.0, work / max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def origins(n):
    if n <= TILE:
        return [0]
    return list(range(0, n - TILE, STRIDE)) + [n - TILE]


def tile_labels(polys, x0, y0, w, h):
    """Clip every polygon to the tile; return YOLO-seg lines."""
    lines = []
    for p in polys:
        p = np.array(p, np.int32)
        bx0, by0 = p.min(0)
        bx1, by1 = p.max(0) + 1
        if bx1 <= x0 or by1 <= y0 or bx0 >= x0 + w or by0 >= y0 + h:
            continue
        # rasterise the object locally, then crop it to the tile
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cv2.fillPoly(m, [p - [bx0, by0]], 1)
        full = m.sum()
        cx0, cy0 = max(bx0, x0), max(by0, y0)
        cx1, cy1 = min(bx1, x0 + w), min(by1, y0 + h)
        crop = m[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
        if full == 0 or crop.sum() < MIN_VISIBLE * full:
            continue
        cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        if len(c) < 3:
            continue
        c = (c + [cx0 - x0, cy0 - y0]) / [w, h]
        lines.append("0 " + " ".join(f"{v:.5f}" for v in c.reshape(-1)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", default="labels_consensus")
    ap.add_argument("--out", default="datasets/challenge_seg")
    ap.add_argument("--folder", help="use only images from this dataset folder")
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for s in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", s))
        os.makedirs(os.path.join(args.out, "labels", s))
    os.makedirs(os.path.join(args.out, "test_full"))

    split, stats, test_rows = {"train": [], "test": []}, {"train": [0, 0], "val": [0, 0]}, []
    for jf in sorted(glob.glob(os.path.join(args.labels, "*.json"))):
        lab = json.load(open(jf))
        name = lab["image"]
        if args.folder and not name.startswith(args.folder + "/"):
            continue
        polys = [o["polygon"] for o in lab["objects"]]
        img = load(os.path.join(args.data, name))
        stem = name.replace("/", "__")[:-4]
        part = "test" if name in TEST else "train"
        split[part].append(name)
        if part == "test":
            cv2.imwrite(os.path.join(args.out, "test_full", stem + ".jpg"), img)
            test_rows.append({"image": name, "label_count": len(polys)})
            continue
        H, W = img.shape[:2]
        for y0 in origins(H):
            for x0 in origins(W):
                lines = tile_labels(polys, x0, y0, min(TILE, W), min(TILE, H))
                if not lines and np.random.rand() > 0.15:   # keep ~15 % empty tiles
                    continue
                tname = f"{stem}_{y0}_{x0}"
                sub = "val" if np.random.rand() < VAL_TILE_FRAC else "train"
                cv2.imwrite(os.path.join(args.out, "images", sub, tname + ".jpg"),
                            img[y0:y0 + TILE, x0:x0 + TILE])
                with open(os.path.join(args.out, "labels", sub, tname + ".txt"), "w") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))
                stats[sub][0] += 1
                stats[sub][1] += len(lines)

    with open(os.path.join(args.out, "test_full_labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "label_count"])
        w.writeheader()
        w.writerows(test_rows)
    json.dump(split, open(os.path.join(args.out, "split.json"), "w"), indent=1)
    with open(os.path.join(args.out, "data.yaml"), "w") as fh:
        fh.write("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: object\n")
    print(f"train: {len(split['train'])} images -> {stats['train'][0]} train tiles "
          f"({stats['train'][1]} instances) + {stats['val'][0]} val tiles ({stats['val'][1]} instances)")
    print(f"test: {len(split['test'])} full images (held out)")


if __name__ == "__main__":
    np.random.seed(0)
    main()
