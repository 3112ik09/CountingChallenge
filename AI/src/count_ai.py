"""
Count objects and draw masks with the fine-tuned YOLO26-seg model.

Per image
---------
1. Resize so the long side is 2048 px (the same scale as training).
2. Sliced inference: 640x640 tiles with 160 px overlap.
3. YOLO26-seg on each tile -> instance masks + confidences.
4. De-duplication: detections near a tile's border are dropped (the
   neighbouring tile sees them whole), then a global NMS in full-image
   coordinates removes overlapping and truncated (contained) duplicates.
5. Count = number of kept instances. Each instance's mask is pasted into a
   full-size label image for the overlay.
6. Writes overlays and predictions.csv (same format as the Non-AI solution).

Usage:
    python src/count_ai.py --weights weights/best.pt --data <dataset_root> --out results
    python src/count_ai.py --weights weights/best.pt --images <folder of jpgs> --out results
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

WORK_SIZE = 2048
TILE, OVERLAP = 640, 160


def load(path, work=WORK_SIZE):
    img = cv2.imread(path)
    s = min(1.0, work / max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def origins(n):
    if n <= TILE:
        return [0]
    return list(range(0, n - TILE, TILE - OVERLAP)) + [n - TILE]


def owned(x0, n, size):
    m = OVERLAP // 2
    return (0 if x0 == 0 else m), (size if x0 + size >= n else size - m)


def global_nms(boxes, scores, iou_thr=0.5, contain_thr=0.7):
    """Suppress duplicates across tiles in full-image coordinates.

    A box is a duplicate if it overlaps a higher-scoring box with IoU > iou_thr,
    or if most of it (> contain_thr of its area) lies inside a kept box. The
    second rule removes truncated copies of an object cut by a tile edge.
    """
    order = np.argsort(-scores)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    for i in order:
        ok = True
        for j in keep:
            ix = max(0, min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0]))
            iy = max(0, min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1]))
            inter = ix * iy
            if inter == 0:
                continue
            if inter / (area[i] + area[j] - inter) > iou_thr or inter / area[i] > contain_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def count_image(model, img, conf=0.25, device=None):
    """Sliced inference -> global de-duplication -> (count, label image)."""
    H, W = img.shape[:2]
    boxes, scores, masks = [], [], []
    for y0 in origins(H):
        for x0 in origins(W):
            t = img[y0:y0 + TILE, x0:x0 + TILE]
            th, tw = t.shape[:2]
            r = model.predict(t, imgsz=TILE, conf=conf, max_det=1000, retina_masks=True,
                              verbose=False, device=device)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            ms = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else None
            # pre-filter: drop detections whose centre is outside this tile's
            # central region (they are seen better by the neighbouring tile)
            ylo, yhi = owned(y0, H, th)
            xlo, xhi = owned(x0, W, tw)
            for k, b in enumerate(xy):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if not (xlo - OVERLAP // 4 <= cx < xhi + OVERLAP // 4 and
                        ylo - OVERLAP // 4 <= cy < yhi + OVERLAP // 4):
                    continue
                bx0, by0, bx1, by1 = b.astype(int)
                boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
                scores.append(sc[k])
                m = ms[k][by0:by1, bx0:bx1] if ms is not None else None
                masks.append((x0 + bx0, y0 + by0, m))
    seg = np.zeros((H, W), np.int32)
    if not boxes:
        return 0, seg
    keep = global_nms(np.array(boxes), np.array(scores))
    for n, i in enumerate(sorted(keep, key=lambda i: -scores[i]), 1):
        gx, gy, m = masks[i]
        if m is None or m.size == 0:
            continue
        region = seg[gy:gy + m.shape[0], gx:gx + m.shape[1]]
        m = m[:region.shape[0], :region.shape[1]]
        region[m & (region == 0)] = n
    return len(keep), seg


def overlay(img, seg, count):
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, size=(seg.max() + 1, 3)).astype(np.uint8)
    colors[0] = 0
    out = img.copy()
    fg = seg > 0
    out[fg] = (0.45 * img[fg] + 0.55 * colors[seg][fg]).astype(np.uint8)
    edges = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    out[edges > 0] = (0, 0, 255)
    txt = f"Count: {count}"
    cv2.rectangle(out, (10, 10), (20 + 28 * len(txt), 75), (0, 0, 0), -1)
    cv2.putText(out, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", help="dataset root (folders of .jpg)")
    ap.add_argument("--images", help="a single folder of .jpg")
    ap.add_argument("--out", default="results")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    if args.data:
        files = sorted(glob.glob(os.path.join(args.data, "*", "*.jpg")))  # one level: image folders only
        names = [f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}" for f in files]
    else:
        files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
        names = [os.path.basename(f)[:-4].replace("__", "/") + ".jpg" for f in files]
    rows = []
    for f, name in zip(files, names):
        img = load(f)
        n, seg = count_image(model, img, args.conf, args.device)
        cv2.imwrite(os.path.join(args.out, name.replace("/", "__")), overlay(img, seg, n))
        rows.append({"image": name, "predicted": n})
        print(f"{name:45s} count={n:4d}", flush=True)
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "predicted"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
