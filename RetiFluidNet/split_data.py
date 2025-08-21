import os
import glob
import json
from sklearn.model_selection import KFold

cirrus_cases     = sorted(glob.glob("RetouchData/Cirrus/retouch_data/*"))
spectralis_cases = sorted(glob.glob("RetouchData/Spectralis/retouch_data/*"))

SEED = 3407
kf = KFold(n_splits=4, shuffle=True, random_state=SEED)

folds = []
# Split both devices in lock-step so fold indices match
for (train_idx_c, val_idx_c), (train_idx_s, val_idx_s) in zip(
        kf.split(cirrus_cases),
        kf.split(spectralis_cases)
    ):

    # uild case lists for this fold
    train_cases = [cirrus_cases[i]   for i in train_idx_c] + [spectralis_cases[i] for i in train_idx_s]
    val_cases   = [cirrus_cases[i]   for i in val_idx_c] + [spectralis_cases[i] for i in val_idx_s]

    # Sanity check
    assert len(train_idx_c) == 18 and len(train_idx_s) == 18, "Train must have 18 per device"
    assert len(val_idx_c)   ==  6 and len(val_idx_s)   ==  6, "Val must have 6 per device"

    # Expand to image/mask paths
    def gather_pngs(cases, suffix):
        imgs = []
        for case in cases:
            base_dir = case if suffix == "oct" else case.replace("retouch_data", "retouch_data_mask")
            imgs += glob.glob(os.path.join(base_dir, f"*_{suffix}_*.png"))
        return sorted(imgs)

    train_images = gather_pngs(train_cases, "oct")
    train_masks  = gather_pngs(train_cases, "mask")
    val_images   = gather_pngs(val_cases,   "oct")
    val_masks    = gather_pngs(val_cases,   "mask")

    # Record fold
    folds.append({
        "train": {
            "cases":     sorted(os.path.basename(c) for c in train_cases),
            "images":    train_images,
            "masks":     train_masks,
            "num_cases": len(train_cases),
            "num_images":len(train_images),
        },
        "val": {
            "cases":     sorted(os.path.basename(c) for c in val_cases),
            "images":    val_images,
            "masks":     val_masks,
            "num_cases": len(val_cases),
            "num_images":len(val_images),
        }
    })

with open("retouch_splitted_4.json", "w") as f:
    json.dump(folds, f, indent=2)

print("Saved stratified K-fold splits to retouch_splitted.json")
