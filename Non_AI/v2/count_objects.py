"""
Non-AI object counting - v2 (classical OpenCV pipeline).

Changes vs v1 (full explanation in README.md):
  [V2-1] Field-of-view mask: dark vignette corners are excluded, so corner
         artefacts are no longer counted as objects.
  [V2-2] Unit area from the left-most mode of the log-area histogram
         (single objects), instead of an iterative median that drifts up when
         touching pairs dominate.
  [V2-3] Round-object ("disc") model: head-on screws are round and ~1.45x the
         area of a side-view screw. They are detected inside clusters from
         distance-transform cores and counted separately, so they no longer
         inflate the area-based count.
  [V2-4] Nut holes are validated more strictly (circularity, size band and a
         ring-of-metal test), and hole-less leftovers are only counted when
         they are clearly nut-sized.
  [V2-5] Speck filter lowered to 0.25u.
  [V2-6] Crowding factor in disc scenes: screws leaning on round heads tilt and
         show more of their head, so inside clusters each takes ~1.1u.

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

# [V2-1] field of view
FOV_BG_FRAC = 0.55         # background darker than 55 % of median => outside FOV
FOV_MIN_INSIDE = 0.70      # blob must have >= 70 % of its pixels inside the FOV

# nut holes / router
HOLE_CIRC_MIN = 0.75       # 4*pi*A/P^2 of a nut hole (circle = 1)
HOLE_AREA_BAND = (0.4, 2.0)  # x median hole area
HOLE_RING_MIN = 0.85       # [V2-4] fraction of a ring around the hole that must be metal
NUT_SCENE_MIN_HOLES = 20
NUT_HOLE_MIN_AREA = 50     # px; screw-head slots are ~20 px
NUT_RESIDUAL_MIN = 0.6     # [V2-4] hole-less leftover must be >= 0.6 nut to count

# blob scenes
SPECK_FRAC = 0.25          # [V2-5] blobs < 0.25u are noise
DISC_CIRC_MIN = 0.85       # [V2-3] singleton round object
DISC_ASPECT_MAX = 1.25
DISC_MIN_SINGLETONS = 3
DISC_CORE_FRAC = 0.8       # distance >= 0.8 r  => inside a disc core
CLUSTER_C = 1.10           # [V2-6] area per object inside clusters, x u (disc scenes);
                           # chosen by leave-one-image-out on the 6 mixed images (c=1.10 in every fold)


# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------
def load(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = WORK_SIZE / max(h, w)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def segment(img):
    """Return (binary foreground mask, field-of-view mask)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    norm = np.clip(g / (bg + 1e-6) * 255, 0, 255).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (5, 5), 0)
    _, m = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # [V2-1] field of view: where the estimated background itself is dark
    # (vignette corners / lens shadow) nothing can be measured reliably.
    fov = (bg > FOV_BG_FRAC * np.median(bg)).astype(np.uint8)
    fov = cv2.erode(fov, np.ones((25, 25), np.uint8))
    return m, fov


def shape_features(comp):
    """Area, circularity, min-area-rect aspect ratio of a binary crop."""
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    a = float(comp.sum())
    per = cv2.arcLength(c, True)
    circ = 4 * np.pi * a / (per * per + 1e-6)
    (_, (rw, rh), _) = cv2.minAreaRect(c)
    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    return a, circ, aspect


# --------------------------------------------------------------------------
# [V2-2] unit area
# --------------------------------------------------------------------------
def unit_area(areas):
    """Area of ONE object = left-most strong mode of the log-area histogram.

    Singletons are the smallest common blob size; touching pairs/triples form
    modes at 2x, 3x. Taking the left-most peak (>= 35 % of the highest peak)
    avoids the upward drift of a plain median when clumps dominate.
    """
    areas = np.asarray(areas, np.float64)
    if len(areas) < 8:                      # too few blobs for a histogram
        return float(np.median(areas[areas <= np.percentile(areas, 50)]))
    la = np.log(areas)
    hist, edges = np.histogram(la, bins=40)
    sm = np.convolve(hist, [1, 2, 3, 2, 1], mode="same").astype(float)
    thr = 0.35 * sm.max()
    peak = int(np.argmax(sm))
    for i in range(len(sm)):
        left = sm[i - 1] if i > 0 else -1
        right = sm[i + 1] if i < len(sm) - 1 else -1
        if sm[i] >= thr and sm[i] >= left and sm[i] >= right:
            peak = i
            break
    p = np.exp(0.5 * (edges[peak] + edges[peak + 1]))
    sel = areas[(areas > 0.7 * p) & (areas < 1.4 * p)]
    return float(np.median(sel)) if len(sel) else float(p)


# --------------------------------------------------------------------------
# Nut branch
# --------------------------------------------------------------------------
def ring_metal_fraction(m, cx, cy, r, n=36):
    """Fraction of points on a circle of radius r around (cx, cy) that are metal."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip((cx + r * np.cos(t)).astype(int), 0, m.shape[1] - 1)
    ys = np.clip((cy + r * np.sin(t)).astype(int), 0, m.shape[0] - 1)
    return float((m[ys, xs] > 0).mean())


def nut_holes(m, fov):
    """Validated nut holes -> (label image, count, median hole area, filled mask)."""
    filled = ndi.binary_fill_holes(m > 0)
    holes = (filled & ~(m > 0)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(holes, connectivity=4)
    cand = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 15:
            continue
        x, y, w, h = st[i, :4]
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        _, circ, _ = shape_features(comp)
        cx, cy = cen[i]
        if circ > HOLE_CIRC_MIN and fov[int(cy), int(cx)]:
            cand.append((i, a, cx, cy))
    if not cand:
        return np.zeros_like(lab), 0, 0.0, filled
    med = float(np.median([a for _, a, _, _ in cand]))
    keep = []
    for i, a, cx, cy in cand:
        if not (HOLE_AREA_BAND[0] * med < a < HOLE_AREA_BAND[1] * med):
            continue
        # [V2-4] a real nut hole is surrounded by a closed ring of metal
        r_h = np.sqrt(a / np.pi)
        if ring_metal_fraction(m, cx, cy, 1.6 * r_h) < HOLE_RING_MIN:
            continue
        keep.append(i)
    out = np.zeros_like(lab)
    for k, i in enumerate(keep, 1):
        out[lab == i] = k
    return out, len(keep), med, filled


def nut_area(filled, hole_lab):
    """Area of one nut (hole included) from isolated blobs holding exactly one hole."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
    # which blob each validated hole sits in -> number of holes per blob
    hole_ids = np.arange(1, hole_lab.max() + 1)
    pos = ndi.labeled_comprehension(lab, hole_lab, hole_ids, lambda v: v[0], int, 0)
    holes_per_blob = np.bincount(pos, minlength=n)
    singles = [st[i, cv2.CC_STAT_AREA] for i in range(1, n) if holes_per_blob[i] == 1]
    return float(np.median(singles)) if len(singles) >= 3 else None


def count_nuts(hole_lab, n_holes, filled, fov):
    """One validated hole = one nut.

    Masks: every pixel of the (hole-filled) foreground is assigned to its
    nearest validated hole (Voronoi), but only within one nut radius, so a
    nut's mask can never stretch across the pile.
    """
    unit = nut_area(filled, hole_lab)
    if unit is None:                                  # fallback: hole area x 6
        unit = 6.0 * float(np.median(np.bincount(hole_lab.ravel())[1:]))
    radius = 1.1 * np.sqrt(unit / np.pi)
    dist, (iy, ix) = ndi.distance_transform_edt(hole_lab == 0, return_indices=True)
    seg = hole_lab[iy, ix]
    seg[(~filled) | (dist > radius)] = 0
    # [V2-4] hole-less leftovers count only when clearly nut-sized
    orphan = (filled & (seg == 0) & (fov > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(orphan, connectivity=8)
    extra, next_id = 0, seg.max() + 1
    for i in range(1, n):
        k = int(st[i, cv2.CC_STAT_AREA] / unit + (1 - NUT_RESIDUAL_MIN))
        if k >= 1:
            extra += k
            seg[lab == i] = next_id
            next_id += 1
    return n_holes + extra, seg, unit


# --------------------------------------------------------------------------
# Blob branch
# --------------------------------------------------------------------------
def split_component(comp_mask, k):
    """Split one blob into k masks: k-means seeds + distance-transform watershed."""
    ys, xs = np.nonzero(comp_mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, km_labels, centres = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    markers = np.zeros(comp_mask.shape, np.int32)
    for i, (cx, cy) in enumerate(centres, 1):
        j = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        markers[ys[j], xs[j]] = i
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
    ws = watershed(-dist, markers, mask=comp_mask > 0)
    # Watershed can starve a seed (a region of a few pixels) in dense
    # clusters. Then fall back to the plain k-means partition, which always
    # gives k regions of similar size. (Affects masks only, not the count.)
    sizes = np.bincount(ws.ravel(), minlength=k + 1)[1:]
    if len(sizes) < k or sizes.min() < 0.25 * len(xs) / k:
        ws = np.zeros(comp_mask.shape, np.int32)
        ws[ys, xs] = km_labels.ravel() + 1
    return ws


def disc_cores(comp, r):
    """[V2-3] Number of round-object cores inside a blob.

    Inside a disc of radius r the distance to the edge reaches r, while a
    side-view screw is only ~half a disc wide. Pixels with distance >=
    0.8 r therefore only exist in the middle of round objects; each connected
    core of sufficient size is one disc.
    """
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    core = (dist >= DISC_CORE_FRAC * r).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    ideal = np.pi * ((1 - DISC_CORE_FRAC) * r) ** 2
    return int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.4 * ideal))


def count_blobs(m, fov):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a <= 30:
            continue
        x, y, w, h = stats[i, :4]
        crop = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        # [V2-1] drop blobs that lie mostly outside the field of view
        if fov[y:y + h, x:x + w][crop > 0].mean() < FOV_MIN_INSIDE:
            continue
        comps.append((i, crop, *shape_features(crop)))
    if not comps:
        return 0, np.zeros(m.shape, np.int32), 0.0

    # [V2-3] round singletons define the disc model (if present)
    is_disc = [c >= DISC_CIRC_MIN and ar <= DISC_ASPECT_MAX for _, _, _, c, ar in comps]
    other = [a for (_, _, a, _, _), d in zip(comps, is_disc) if not d]
    u = unit_area(other if len(other) >= 3 else [c[2] for c in comps])
    disc_areas = [a for (_, _, a, _, _), d in zip(comps, is_disc) if d and a > 0.6 * u]
    has_discs = len(disc_areas) >= DISC_MIN_SINGLETONS
    a_d = float(np.median(disc_areas)) if has_discs else 0.0
    r_d = np.sqrt(a_d / np.pi) if has_discs else 0.0

    seg = np.zeros(m.shape, np.int32)
    total, next_id = 0, 1
    for (i, crop, a, circ, ar), d in zip(comps, is_disc):
        if a < SPECK_FRAC * u:
            continue
        if d and has_discs and a < 1.6 * a_d:
            k = 1                                   # a single round object
        else:
            n_d = disc_cores(crop, r_d) if has_discs else 0
            rest = max(0.0, a - n_d * a_d)
            c = CLUSTER_C if (has_discs and a > 2.5 * u) else 1.0
            k = max(1, n_d + int(round(rest / (u * c))))
        crop_p = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
        ws = split_component(crop_p, k)[2:-2, 2:-2] if k > 1 else crop.astype(np.int32)
        total += k
        x, y, w, h = stats[i, :4]
        region = seg[y:y + h, x:x + w]
        region[ws > 0] = ws[ws > 0] + next_id - 1
        next_id += ws.max()
    return total, seg, u


# --------------------------------------------------------------------------
def count_image(img):
    m, fov = segment(img)
    hole_lab, n_holes, hole_med, filled = nut_holes(m, fov)
    if n_holes >= NUT_SCENE_MIN_HOLES and hole_med >= NUT_HOLE_MIN_AREA:
        total, seg, u = count_nuts(hole_lab, n_holes, filled, fov)
        return total, seg, u, "nuts"
    total, seg, u = count_blobs(m, fov)
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
