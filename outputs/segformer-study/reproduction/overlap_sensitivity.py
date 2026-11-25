"""Compare all nine trained models on UIIS after screening potential generator-source matches."""
import argparse
import csv
from pathlib import Path

import numpy as np

ARMS = ("synthetic", "mixed", "real")
SEEDS = (17, 29, 43)
CLASSES = ("background", "wrecks", "fish", "reefs", "aquatic plants", "divers", "robots")
N_BOOT = 2000


def save(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def read_all(study):
    ids, groups = None, None
    arms = []
    for arm in ARMS:
        seeds = []
        for seed in SEEDS:
            stats = study / "metrics" / arm / f"seed-{seed}" / "uiis_test" / "confusions.npz"
            with np.load(stats) as data:
                if ids is None:
                    ids, groups = data["id"], data["group"]
                assert np.array_equal(ids, data["id"]) and np.array_equal(groups, data["group"])
                seeds.append(data["cm"])
        arms.append(seeds)
    return np.array(arms), ids, groups


def iou(cm):
    diagonal = cm.diagonal(axis1=-2, axis2=-1)
    union = cm.sum(axis=-2) + cm.sum(axis=-1) - diagonal
    return np.divide(diagonal, union, out=np.full(diagonal.shape, np.nan), where=union > 0)


def row_value(x):
    return round(float(x), 7) if np.isfinite(x) else ""


def main(args):
    output = args.study / "metrics"
    all_cm, ids, all_groups = read_all(args.study)
    with args.flags.open(newline="") as handle:
        flags = {r["id"]: r for r in csv.DictReader(handle) if r["split"] == "uiis_test"}
    assert set(flags) == set(ids)
    totals, deltas = [], []
    for name, criterion in [("no_UIIS_train_match", "matched_uiis"), ("no_known_generator_source_match", "known_overlap")]:
        keep = np.array([flags[idx][criterion] == "0" for idx in ids])
        cm = all_cm[:, :, keep]
        groups = all_groups[keep]
        group_names, inverse = np.unique(groups, return_inverse=True)
        per_group = np.stack([cm[:, :, inverse == index].sum(axis=2) for index in range(len(group_names))])
        class_support = (cm[0, 0].sum(axis=-1) > 0).sum(axis=0)
        pooled_iou = iou(cm.sum(axis=2))
        rng = np.random.default_rng(1082)
        weights = rng.multinomial(len(group_names), [1 / len(group_names)] * len(group_names), size=N_BOOT)
        sampled = (weights.astype(float) @ per_group.reshape(len(group_names), -1)).reshape((N_BOOT,) + per_group.shape[1:])
        sampled_iou = iou(sampled)
        # The six-class estimate requires all six reference classes. Report bootstrap retention.
        has_classes = (sampled[:, 0, 0].sum(axis=-1)[:, 1:] > 0).all(axis=1)
        common = dict(subset=name, images=len(groups), visual_groups=len(group_names))
        for idx, label in enumerate(("foreground_mIoU",) + CLASSES[1:]):
            if idx == 0:
                point = pooled_iou[..., 1:].mean(axis=-1)
                if not (class_support[1:] > 0).all():
                    point = np.full(point.shape, np.nan)
                boot = sampled_iou[..., 1:].mean(axis=-1)
                valid = has_classes
                support = ""
            else:
                point, boot = pooled_iou[..., idx], sampled_iou[..., idx]
                valid = sampled[:, 0, 0, idx].sum(axis=-1) > 0
                support = int(class_support[idx])
                if not support:
                    point = np.full(point.shape, np.nan)
            metric = label if idx == 0 else label + "_IoU"
            for a, arm in enumerate(ARMS):
                totals.append(dict(**common, arm=arm, metric=metric, reference_images_with_class=support,
                                   seed_mean_percent=row_value(100 * point[a].mean()),
                                   seed_sd_pp=row_value(100 * point[a].std(ddof=1)),
                                   **{f"seed_{seed}_percent":row_value(100 * point[a, s]) for s,seed in enumerate(SEEDS)}))
            for a, b in ((1, 2), (0, 2), (1, 0)):
                pair = 100 * (point[a] - point[b])
                differences = 100 * (boot[:, a] - boot[:, b]).mean(axis=1)
                valid_boot = differences[valid & np.isfinite(differences)]
                lo, hi = np.quantile(valid_boot, [.025,.975]) if len(valid_boot) else (np.nan, np.nan)
                deltas.append(dict(**common, metric=metric, arm=ARMS[a], reference_arm=ARMS[b],
                                   delta_pp=row_value(pair.mean()), lower95_pp=row_value(lo), upper95_pp=row_value(hi),
                                   paired_seed_sd_pp=row_value(pair.std(ddof=1)), bootstrap_draws=N_BOOT,
                                   draws_with_required_reference_classes=len(valid_boot)))
    save(output / "uiis-overlap-sensitivity.csv", totals)
    save(output / "uiis-overlap-paired-deltas.csv", deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--flags", type=Path, required=True)
    main(parser.parse_args())
