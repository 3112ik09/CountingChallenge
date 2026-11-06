"""
Non-AI object counting (classical OpenCV pipeline).

Pipeline per image (full explanation in README.md)
------------------
1. Resize so the long side is WORK_SIZE px.
2. Illumination flattening: divide by a background estimate
   (grey closing + Gaussian blur). Removes vignetting / uneven light.
3. Otsu threshold (objects darker than background) + 3x3 opening.
4. Hole analysis: holes = fill_holes(mask) - mask; keep round, nut-sized ones.
5. Scene router: many round holes -> nut branch, else blob branch.
6A. Nut branch: count = number of holes (1 hole = 1 nut); masks by watershed
    seeded at the holes; hole-less blobs counted by area.
6B. Blob branch: per-image unit area u (iterative median); count per blob =
    round(area / u); masks by k-means seeds + distance-transform watershed.
7. Overlay coloured masks + count; write predictions.csv.

Usage
-----
    python count_objects.py --data <dataset_root> --out results
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

WORK_SIZE = 1600
CIRCULARITY_MIN = 0.7      # 4*pi*A/P^2: 1 = circle; nut holes ~0.8-0.9
NUT_SCENE_MIN_HOLES = 20   # router: this many round holes => nut scene
NUT_HOLE_MIN_AREA = 50     # px at WORK_SIZE; screw-head slots are ~20 px


def load(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = WORK_SIZE / max(h, w)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def foreground_mask(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    norm = np.clip(g / (bg + 1e-6) * 255, 0, 255).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (5, 5), 0)
    _, m = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # NOTE: holes are deliberately NOT filled - nut centres and the gaps
    # between touching objects must not be counted as object area.
    return m


def unit_area(areas):
    """Area of a single object: iterate median restricted to [0.6u, 1.4u]."""
    u = np.percentile(areas, 25)
    for _ in range(10):
        sel = areas[(areas > 0.6 * u) & (areas < 1.4 * u)]
        if len(sel) == 0:
            break
        u_new = float(np.median(sel))
        if abs(u_new - u) < 1:
            break
        u = u_new
    return u


def split_component(comp_mask, k):
    """Split one connected blob into k object masks.

    Seeds = k-means centres of the blob's pixel coordinates, then a marker
    watershed on the distance transform snaps the boundaries to the narrow
    "necks" between touching objects.
    """
    ys, xs = np.nonzero(comp_mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, _, centres = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    markers = np.zeros(comp_mask.shape, np.int32)
    for i, (cx, cy) in enumerate(centres, 1):
        # snap the seed onto the nearest foreground pixel
        j = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        markers[ys[j], xs[j]] = i
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
    return watershed(-dist, markers, mask=comp_mask > 0)


def nut_holes(m):
    """Find nut holes: enclosed background regions that are round.

    holes = fill_holes(mask) - mask. Gaps enclosed between touching nuts are
    also holes, but they are irregular (low circularity) and get rejected.
    Returns a label image of accepted holes and their count.
    """
    filled = ndi.binary_fill_holes(m > 0)
    holes = (filled & ~(m > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    cand = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 15:
            continue
        x, y, w, h = st[i, :4]
        cnt, _ = cv2.findContours((lab[y:y + h, x:x + w] == i).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = cv2.arcLength(cnt[0], True)
        circ = 4 * np.pi * a / (per * per + 1e-6)
        if circ > CIRCULARITY_MIN:
            cand.append((i, a))
    if not cand:
        return np.zeros_like(lab), 0, 0.0, filled
    med = float(np.median([a for _, a in cand]))
    keep = [i for i, a in cand if 0.4 * med < a < 2.2 * med]
    out = np.zeros_like(lab)
    for k, i in enumerate(keep, 1):
        out[lab == i] = k
    return out, len(keep), med, filled


def count_nuts(m, hole_lab, n_holes, filled):
    """Nut scene: one round hole = one nut. Masks by watershed from the holes."""
    filled_u8 = filled.astype(np.uint8)
    dist = cv2.distanceTransform(filled_u8, cv2.DIST_L2, 5)
    seg = watershed(-dist, hole_lab, mask=filled)
    # blobs that contain no visible hole (e.g. nut lying on its side):
    # fall back to area / median nut area
    sizes = np.bincount(seg.ravel())[1:]
    unit = float(np.median(sizes)) if len(sizes) else 1.0
    orphan = (filled & (seg == 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(orphan, connectivity=8)
    extra, next_id = 0, seg.max() + 1
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        k = int(round(a / unit))
        if k >= 1:
            extra += k
            seg[lab == i] = next_id
            next_id += 1
    return n_holes + extra, seg, unit


def count_image(img):
    m = foreground_mask(img)
    hole_lab, n_holes, hole_med, filled = nut_holes(m)
    if n_holes >= NUT_SCENE_MIN_HOLES and hole_med >= NUT_HOLE_MIN_AREA:
        total, seg, u = count_nuts(m, hole_lab, n_holes, filled)
        return total, seg, u, "nuts"
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    valid = areas[areas > 30]  # ignore dust when estimating the unit area
    u = unit_area(valid)
    min_area = 0.3 * u  # anything smaller than 30 % of an object is noise

    seg = np.zeros(m.shape, np.int32)
    total, next_id = 0, 1
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        x, y, w, h = stats[i, :4]
        crop = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        crop = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
        # area-based count: a blob of area A contains ~A/u objects
        est = max(1, int(round(a / u)))
        if est > 1:
            ws = split_component(crop, est)[2:-2, 2:-2]
        else:
            ws = crop[2:-2, 2:-2].astype(np.int32)
        total += est
        region = seg[y:y + h, x:x + w]
        region[ws > 0] = ws[ws > 0] + next_id - 1
        next_id += ws.max()
    return total, seg, u, "blobs"


def overlay(img, seg, count):
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, size=(seg.max() + 1, 3)).astype(np.uint8)
    colors[0] = 0
    col = colors[seg]
    out = img.copy()
    fg = seg > 0
    out[fg] = (0.45 * img[fg] + 0.55 * col[fg]).astype(np.uint8)
    edges = cv2.morphologyEx((seg > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    out[edges > 0] = (0, 0, 255)
    txt = f"Count: {count}"
    cv2.rectangle(out, (10, 10), (20 + 28 * len(txt), 75), (0, 0, 0), -1)
    cv2.putText(out, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (folders of .jpg)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.data, "*", "*.jpg"))):
        img = load(f)
        count, seg, u, mode = count_image(img)
        name = f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"
        cv2.imwrite(os.path.join(args.out, name.replace("/", "__")), overlay(img, seg, count))
        rows.append({"image": name, "predicted": count, "mode": mode})
        print(f"{name:45s} count={count:4d}  mode={mode:5s} unit_area={u:.0f}")
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "predicted", "mode"])
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
