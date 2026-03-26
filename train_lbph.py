"""
train_lbph.py – Train an LBPH face recogniser on the VN-celeb dataset.

Dataset structure expected:
    VN-celeb/
        <id>/          # folder name is the integer label (1, 2, … 1020)
            0.png
            10.png
            ...

Output:
    custom_models/lbph_model.xml   – LBPH recogniser (load with rec.read())
    custom_models/lbph_model.json  – {label_id: folder_name} mapping

Usage:
    python train_lbph.py [--dataset VN-celeb] [--out custom_models]
                         [--max-per-id 50] [--img-size 100]
                         [--min-images 5]
"""

import os
import glob
import json
import argparse
import time

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DATASET_DIR = "VN-celeb"
OUT_DIR     = "custom_models"
IMG_SIZE    = (100, 100)   # LBPH works well at 100×100

# ──────────────────────────────────────────────────────────────────────────────
# Haar detector (same as ws_server.py) – used to crop face ROIs
# ──────────────────────────────────────────────────────────────────────────────

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(gray: np.ndarray):
    """Return the largest detected face as (x,y,w,h) or None."""
    dets = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(dets) == 0:
        return None
    # Pick the largest box (most likely the subject)
    return max(dets, key=lambda b: b[2] * b[3])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_dir: str, max_per_id: int, img_size: tuple,
                 min_images: int) -> tuple[list, list, dict]:
    """
    Walk dataset_dir/<id>/*.png, detect face, resize → img_size gray.

    Returns:
        all_images – list of np.ndarray (all accepted face ROIs)
        all_labels – list of int (compact 0-indexed label per ROI)
        label_map  – {int_label: str(folder_name)}
    """
    all_images: list = []
    all_labels: list = []
    label_map: dict  = {}

    identity_dirs = sorted(
        [d for d in os.listdir(dataset_dir)
         if os.path.isdir(os.path.join(dataset_dir, d))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    total_ids   = len(identity_dirs)
    skipped_ids = 0
    t0 = time.time()
    print(f"Found {total_ids} identity folders.")

    for idx, folder in enumerate(identity_dirs):
        folder_path = os.path.join(dataset_dir, folder)
        img_paths   = sorted(
            glob.glob(os.path.join(folder_path, "*.png")) +
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.jpeg"))
        )

        face_rois = []
        for img_path in img_paths[:max_per_id * 3]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            box  = detect_face(gray)
            if box is None:
                continue
            x, y, w, h = box
            roi = cv2.resize(gray[y:y+h, x:x+w], img_size)
            face_rois.append(roi)
            if len(face_rois) >= max_per_id:
                break

        if len(face_rois) < min_images:
            skipped_ids += 1
            continue

        label_id = len(label_map)
        label_map[label_id] = folder
        all_images.extend(face_rois)
        all_labels.extend([label_id] * len(face_rois))

        if (idx + 1) % 100 == 0 or (idx + 1) == total_ids:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{total_ids}] Accepted: {len(label_map)} ids, "
                  f"Samples: {len(all_images)}, Skipped: {skipped_ids}, "
                  f"Elapsed: {elapsed:.1f}s")

    return all_images, all_labels, label_map


def split_dataset(images: list, labels: list,
                  val_ratio: float = 0.2, seed: int = 42):
    """
    Stratified train/val split: for each identity keep last ceil(n*val_ratio)
    samples as validation, the rest as training.

    Returns train_images, train_labels, val_images, val_labels.
    """
    rng = np.random.default_rng(seed)
    train_imgs, train_lbls, val_imgs, val_lbls = [], [], [], []

    # Group indices by label
    from collections import defaultdict
    idx_by_label: dict = defaultdict(list)
    for i, lbl in enumerate(labels):
        idx_by_label[lbl].append(i)

    for lbl, idxs in sorted(idx_by_label.items()):
        arr = np.array(idxs)
        rng.shuffle(arr)
        n_val = max(1, int(np.ceil(len(arr) * val_ratio)))
        val_idxs   = arr[:n_val]
        train_idxs = arr[n_val:]
        if len(train_idxs) == 0:
            # Not enough samples – use all for train, duplicate last for val
            train_idxs = arr
            val_idxs   = arr[-1:]
        for i in train_idxs:
            train_imgs.append(images[i]); train_lbls.append(labels[i])
        for i in val_idxs:
            val_imgs.append(images[i]);   val_lbls.append(labels[i])

    return train_imgs, train_lbls, val_imgs, val_lbls


# ──────────────────────────────────────────────────────────────────────────────
# Train + save
# ──────────────────────────────────────────────────────────────────────────────

def train_and_save(images: list, labels: list, label_map: dict,
                   out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nTraining LBPH on {len(images)} samples across {len(label_map)} identities…")
    rec = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    rec.train(images, np.array(labels, dtype=np.int32))

    model_path = os.path.join(out_dir, "lbph_model.xml")
    json_path  = os.path.join(out_dir, "lbph_model.json")

    rec.write(model_path)
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    print(f"\nSaved:")
    print(f"  Model : {model_path}")
    print(f"  Labels: {json_path}")


def evaluate(rec, val_images: list, val_labels: list, label_map: dict,
             threshold: float = 80.0) -> dict:
    """
    Evaluate recogniser on validation set.
    Returns dict with overall accuracy + per-identity breakdown.
    """
    from collections import defaultdict
    per_id_correct = defaultdict(int)
    per_id_total   = defaultdict(int)
    correct = total = 0

    for img, true_lbl in zip(val_images, val_labels):
        pred_id, confidence = rec.predict(img)
        # Apply confidence threshold (same as ws_server.py)
        if confidence >= threshold:
            pred_id = -1   # reject → Unknown
        per_id_total[true_lbl] += 1
        total += 1
        if pred_id == true_lbl:
            per_id_correct[true_lbl] += 1
            correct += 1

    accuracy = correct / max(total, 1)

    # Per-identity rows (only show worst 10 for brevity)
    rows = []
    for lbl in sorted(per_id_total):
        n  = per_id_total[lbl]
        ok = per_id_correct[lbl]
        rows.append((lbl, label_map.get(lbl, str(lbl)), ok, n, ok/n))

    rows_sorted = sorted(rows, key=lambda r: r[4])  # ascending accuracy

    return {
        "correct":   correct,
        "total":     total,
        "accuracy":  accuracy,
        "per_id":    rows,
        "worst":     rows_sorted[:10],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train LBPH recogniser on VN-celeb")
    ap.add_argument("--dataset",     default=DATASET_DIR)
    ap.add_argument("--out",         default=OUT_DIR)
    ap.add_argument("--max-per-id",  type=int,   default=50)
    ap.add_argument("--img-size",    type=int,   default=100)
    ap.add_argument("--min-images",  type=int,   default=5)
    ap.add_argument("--val-ratio",   type=float, default=0.2,
                    help="Fraction of each identity's images held out for validation (default 0.2)")
    ap.add_argument("--threshold",   type=float, default=80.0,
                    help="LBPH confidence threshold for Unknown rejection (default 80)")
    args = ap.parse_args()

    img_size = (args.img_size, args.img_size)

    print(f"{'='*62}")
    print(f"  LBPH Training – VN-celeb")
    print(f"{'='*62}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Output    : {args.out}")
    print(f"  Max/ID    : {args.max_per_id}")
    print(f"  Min faces : {args.min_images}")
    print(f"  Val ratio : {args.val_ratio}")
    print(f"  ROI size  : {img_size}\n")

    all_images, all_labels, label_map = load_dataset(
        args.dataset, args.max_per_id, img_size, args.min_images
    )

    if len(all_images) == 0:
        print("No training data collected. Check --dataset path.")
        return

    # ── Train / Val split ────────────────────────────────────────────────
    train_imgs, train_lbls, val_imgs, val_lbls = split_dataset(
        all_images, all_labels, val_ratio=args.val_ratio
    )

    print(f"\nSplit summary:")
    print(f"  Train : {len(train_imgs)} samples")
    print(f"  Val   : {len(val_imgs)} samples")
    print(f"  IDs   : {len(label_map)}")

    # ── Train ─────────────────────────────────────────────────────────────
    train_and_save(train_imgs, train_lbls, label_map, args.out)

    # ── Evaluate ──────────────────────────────────────────────────────────
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(os.path.join(args.out, "lbph_model.xml"))

    print("\nEvaluating on validation set…")
    result = evaluate(rec, val_imgs, val_lbls, label_map, threshold=args.threshold)

    print(f"\n{'='*62}")
    print(f"  Validation Results")
    print(f"{'='*62}")
    print(f"  Total identities : {len(label_map)}")
    print(f"  Val samples      : {result['total']}")
    print(f"  Correct          : {result['correct']}")
    print(f"  Top-1 Accuracy   : {result['accuracy']*100:.2f}%")

    if result['worst']:
        print(f"\n  Worst-10 identities (by val accuracy):")
        print(f"  {'ID':>6}  {'Folder':>10}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
        print(f"  {'-'*45}")
        for lbl, folder, ok, n, acc in result['worst']:
            print(f"  {lbl:>6}  {folder:>10}  {ok:>7}  {n:>5}  {acc*100:>5.1f}%")

    print(f"{'='*62}")
    print(f"\nTraining complete. Use the model with:")
    print(f"  python ws_server.py --recognizer {args.out}/lbph_model.xml")


if __name__ == "__main__":
    main()
