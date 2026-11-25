"""Build study tables, figures, and a standalone HTML report from frozen predictions."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Patch

from evaluate import DATASETS, LABELS, class_iou, read_manifest, safe_divide, scores


ARMS = ["synthetic", "mixed", "real"]
ARM_NAMES = {"synthetic": "Synthetic only", "mixed": "50% synthetic + 50% real", "real": "Real only"}
SPLIT_NAMES = {"uiis_test": "UIIS", "deepfish_test": "DeepFish", "coralscapes_test": "CoralscapesV2"}
SEEDS = [17, 29, 43]
COLORS = ["#725bad", "#068874", "#395c81"]
METRIC_NAMES = {"foreground_mIoU": "Foreground mIoU", "fish_IoU": "Fish IoU", "fish_Dice": "Fish Dice",
                "fish_precision": "Fish precision", "fish_recall": "Fish recall", "negative_fish_area": "False fish area"}
PLOT_PALETTE = np.array([[0, 0, 0], [83, 133, 255], [20, 235, 138], [255, 184, 60], [235, 90, 218],
                         [255, 85, 70], [100, 224, 255]], dtype=np.uint8)
N_BOOT = 2000


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(float(v), 10) if np.isfinite(v) else "") if
                             isinstance(v, (float, np.floating)) else v for k, v in row.items()})


def load_split(out, split):
    cms, negatives, ids, groups = [], [], None, None
    for arm in ARMS:
        arm_cm, arm_neg = [], []
        for seed in SEEDS:
            root = out / "metrics" / arm / f"seed-{seed}" / split
            if not (root / "complete.json").is_file():
                raise FileNotFoundError(f"Evaluation incomplete at {root}")
            with np.load(root / "confusions.npz", allow_pickle=False) as source:
                if ids is None:
                    ids, groups = source["id"], source["group"]
                if not np.array_equal(source["id"], ids) or not np.array_equal(source["group"], groups):
                    raise ValueError(f"Pairing mismatch at {root}")
                arm_cm.append(source["cm"])
                arm_neg.append(source["negative"])
        cms.append(arm_cm)
        negatives.append(arm_neg)
    return np.array(cms), np.array(negatives), ids, groups


def summarize(out, split, cms, neg, ids, groups, rows):
    measures = list(METRIC_NAMES) if split == "uiis_test" else list(METRIC_NAMES)[1:]
    pooled_cm = cms.sum(2)
    pooled = scores(pooled_cm, neg.sum(2))
    labels_iou = class_iou(pooled_cm)
    hit = pooled_cm.diagonal(axis1=-2, axis2=-1)
    reference, predicted = pooled_cm.sum(axis=-1), pooled_cm.sum(axis=-2)
    class_metrics = {"dice": safe_divide(2 * hit, reference + predicted),
                     "precision": safe_divide(hit, predicted), "recall": safe_divide(hit, reference)}
    if split == "uiis_test":
        for metric, values in class_metrics.items():
            name = "foreground_macro_" + metric
            value = values[..., 1:]
            pooled[name] = safe_divide(np.nansum(value, axis=-1), np.isfinite(value).sum(axis=-1))
    for a, arm in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            for metric in measures:
                rows["per_run"].append(dict(dataset=split, arm=arm, seed=seed, metric=metric,
                                            fraction=pooled[metric][a, s]))
        for metric in measures:
            values = pooled[metric][a]
            rows["summary"].append(dict(dataset=split, arm=arm, metric=metric, seed_mean=values.mean(),
                                        seed_sd=values.std(ddof=1)))
        if split == "uiis_test":
            for metric in class_metrics:
                name = "foreground_macro_" + metric
                values = pooled[name][a]
                rows["summary"].append(dict(dataset=split, arm=arm, metric=name, seed_mean=values.mean(),
                                            seed_sd=values.std(ddof=1)))
                for s, seed in enumerate(SEEDS):
                    rows["per_run"].append(dict(dataset=split, arm=arm, seed=seed, metric=name, fraction=values[s]))
            for idx, label in enumerate(LABELS):
                counts = cms[a, 0, :, idx, :].sum(axis=1)
                rows["class_iou"].append(dict(dataset=split, arm=arm, class_id=idx, class_name=label,
                                              images_with_reference=int((counts > 0).sum()),
                                              reference_pixels=int(counts.sum()),
                                              seed_mean_iou=labels_iou[a, :, idx].mean(),
                                              seed_sd_iou=labels_iou[a, :, idx].std(ddof=1),
                                              **{f"seed_mean_{m}": v[a, :, idx].mean()
                                                 for m, v in class_metrics.items()},
                                              **{f"seed_sd_{m}": v[a, :, idx].std(ddof=1)
                                                 for m, v in class_metrics.items()}))

    names, inv = np.unique(groups, return_inverse=True)
    group_cm = np.stack([cms[:, :, inv == g].sum(2) for g in range(len(names))], axis=0)
    group_neg = np.stack([neg[:, :, inv == g].sum(2) for g in range(len(names))], axis=0)
    if split != "uiis_test":
        for g, name in enumerate(names):
            group_stats = scores(group_cm[g], group_neg[g])
            for a, arm in enumerate(ARMS):
                rows["groups"].append(dict(dataset=split, group=name, frames=int((inv == g).sum()),
                                           arm=arm, seed_mean_fish_iou=group_stats["fish_IoU"][a].mean(),
                                           reference_fish_pixels=int(group_cm[g, a, 0, 2].sum())))

    rng = np.random.default_rng(781 + DATASETS.index(split))
    weights = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=N_BOOT).astype(float)
    resampled_cm = (weights @ group_cm.reshape(len(names), -1)).reshape((N_BOOT,) + group_cm.shape[1:])
    resampled_neg = (weights @ group_neg.reshape(len(names), -1)).reshape((N_BOOT,) + group_neg.shape[1:])
    bs_scores = scores(resampled_cm, resampled_neg)
    for a, b in [(1, 2), (0, 2), (1, 0)]:
        for metric in measures:
            deltas = (pooled[metric][a] - pooled[metric][b])
            boot = (bs_scores[metric][:, a] - bs_scores[metric][:, b]).mean(axis=1)
            if metric == "foreground_mIoU":
                # Keep the primary six-class estimand fixed on every bootstrap draw.
                boot = np.where((resampled_cm[:, a, 0, 1:, :].sum(axis=-1) > 0).all(axis=-1), boot, np.nan)
            valid = boot[np.isfinite(boot)]
            lo, hi = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
            rows["differences"].append(dict(dataset=split, arm=ARMS[a], reference_arm=ARMS[b],
                                            metric=metric, difference_pp=100 * deltas.mean(),
                                            lower95_pp=100 * lo, upper95_pp=100 * hi,
                                            paired_seed_sd_pp=100 * deltas.std(ddof=1),
                                            bootstrap_groups=len(names), valid_bootstraps=len(valid),
                                            **{f"seed_{seed}_difference_pp": 100 * deltas[s]
                                               for s, seed in enumerate(SEEDS)}))
    return pooled


def effect_plot(out, lookup):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, split in zip(axes, DATASETS):
        metric = "foreground_mIoU" if split == "uiis_test" else "fish_IoU"
        arr = lookup[split][metric] * 100
        ax.bar(range(3), arr.mean(axis=1), color=COLORS, width=0.67)
        for a in range(3):
            ax.scatter(a + np.linspace(-0.13, 0.13, 3), arr[a], s=23, zorder=3, color="white",
                       edgecolor="#273546", linewidth=0.65)
        ax.set_title(f"{SPLIT_NAMES[split]}\n{METRIC_NAMES[metric]}", fontsize=12)
        ax.set_xticks(range(3), ["Synthetic", "50/50", "Real"])
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("IoU (%)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "training-comparison.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def replacement_plot(out, contrasts):
    baseline = {r["dataset"]: r for r in contrasts if r["arm"] == "mixed" and r["reference_arm"] == "real"
                and r["metric"] == ("foreground_mIoU" if r["dataset"] == "uiis_test" else "fish_IoU")}
    with (out/"metrics/uiis-overlap-paired-deltas.csv").open(newline="") as file:
        screened = next(r for r in csv.DictReader(file) if r["subset"] == "no_known_generator_source_match"
                        and r["metric"] == "foreground_mIoU" and r["arm"] == "mixed" and r["reference_arm"] == "real")
    labels = ["UIIS · full holdout\nforeground mIoU", "UIIS · generator-source screen\nforeground mIoU",
              "DeepFish\nfish IoU", "CoralscapesV2\nfish IoU"]
    records = [baseline["uiis_test"], screened, baseline["deepfish_test"], baseline["coralscapes_test"]]
    estimates = [float(r.get("difference_pp", r.get("delta_pp"))) for r in records]
    endpoints = np.array([[float(r["lower95_pp"]), float(r["upper95_pp"])] for r in records])
    fig, ax = plt.subplots(figsize=(10, 4.15))
    fig.subplots_adjust(left=.30, right=.77, top=.88, bottom=.20)
    ax.axvline(0, color="#4b5563", linewidth=1.2)
    ax.text(1.04, 3.43, "Difference [95% interval]", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=9, color="#485963")
    for i, (value, (lo, hi)) in enumerate(zip(estimates, endpoints)):
        y = 3 - i
        ax.plot([lo, hi], [y, y], "-", lw=2.2, color="#308777", solid_capstyle="round")
        ax.plot([lo, hi], [y, y], "|", color="#308777", markersize=10)
        ax.plot(value, y, "o", ms=7, color="#045749", zorder=3)
        ax.text(1.04, y, f"{display(value, 2, True)}  [{display(lo, 2, True)}, {display(hi, 2, True)}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=10, color="#1f3640")
    ax.set_yticks([3,2,1,0], labels, fontsize=10)
    ax.set_ylim(-.55, 3.60)
    ax.set_xlabel("50/50 minus real only (percentage points)", labelpad=9)
    ax.set_title("Replacing half the real training images with SynTIDE", loc="left", fontsize=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.savefig(out/"figures/replacement-differences.png", dpi=175, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def deepfish_habitat_differences(out, data):
    cms, _, ids, _ = data
    habitat_rows = read_manifest(out/"reproduction/manifests/deepfish-habitats.csv")
    mapping = {row["id"]: row["habitat"] for row in habitat_rows}
    if set(ids) != set(mapping):
        raise ValueError("DeepFish habitat IDs disagree with the frozen evaluation set")
    habitat = np.array([mapping[image_id] for image_id in ids])
    names = np.unique(habitat)
    tp = cms[..., 2, 2]
    union = cms[..., 2, :].sum(axis=-1) + cms[..., :, 2].sum(axis=-1) - tp
    group_tp = np.stack([tp[..., habitat == h].sum(axis=-1) for h in names])
    group_union = np.stack([union[..., habitat == h].sum(axis=-1) for h in names])
    pooled = safe_divide(group_tp.sum(axis=0), group_union.sum(axis=0))
    weights = np.random.default_rng(1832).multinomial(len(names), np.full(len(names), 1/len(names)), size=N_BOOT)
    boot = safe_divide(np.einsum("bh,has->bas", weights, group_tp),
                       np.einsum("bh,has->bas", weights, group_union))
    rows = []
    for a, b in [(1, 2), (0, 2), (1, 0)]:
        diff = 100 * (pooled[a] - pooled[b])
        distribution = 100 * (boot[:,a] - boot[:,b]).mean(axis=1)
        valid = distribution[np.isfinite(distribution)]
        lo, hi = np.quantile(valid, [0.025,0.975]) if len(valid) else (np.nan, np.nan)
        rows.append(dict(arm=ARMS[a], reference_arm=ARMS[b], metric="fish_IoU", images=len(ids),
                         habitats=len(names), difference_pp=diff.mean(), lower95_pp=lo, upper95_pp=hi,
                         bootstrap_draws=N_BOOT, valid_bootstraps=len(valid)))
    write_csv(out/"metrics/deepfish-habitat-sensitivity.csv", rows)
    return rows


def overlay(image, ids, binary):
    if binary:
        ids = np.where(ids == 255, 255, np.where(ids == 2, 2, 0))
    rgb = np.asarray(image).astype(np.float32)
    valid_fg = (ids > 0) & (ids < len(LABELS))
    rgb[valid_fg] = rgb[valid_fg] * 0.28 + PLOT_PALETTE[ids[valid_fg]] * 0.72
    rgb[ids == 255] = rgb[ids == 255] * 0.35 + 160 * 0.65
    return rgb.astype(np.uint8)


def draw_examples(out, manifests, data):
    examples = []
    for split in DATASETS:
        cms, _, ids, _ = data[split]
        rows = read_manifest(manifests / f"{split}.csv")
        order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(("segformer-example-17/" + ids[i]).encode()).digest())
        gt_counts = cms[0, 0].sum(axis=-1)
        # Pick by ground truth and a stable ID hash, independently of model scores.
        if split == "uiis_test":
            strata = [gt_counts[:, 2] > 0, (gt_counts[:, 1:].sum(1) - gt_counts[:, 2]) > 0]
        elif split == "deepfish_test":
            strata = [gt_counts[:, 2] > 0, gt_counts[:, 2] == 0]
        else:
            strata = [gt_counts[:, 2] > 0, gt_counts[:, 2] > 0]
        selected = set()
        used_groups = set()
        for membership in strata:
            possible = [i for i in order if membership[i] and i not in selected]
            if not possible:
                continue
            index = next((i for i in possible if rows[i]["group"] not in used_groups), possible[0])
            selected.add(index)
            used_groups.add(rows[index]["group"])
            examples.append((split, index, rows[index]))
    fig, axes = plt.subplots(len(examples), 5, figsize=(14, len(examples)*1.95), squeeze=False)
    for ax, name in zip(axes[0], ["Image", "Ground truth", "Synthetic only", "50% / 50%", "Real only"]):
        ax.set_title(name, fontsize=12)
    metadata = []
    for r, (split, index, row) in enumerate(examples):
        crop = None
        with Image.open(row["mask"]) as src:
            reference_image = src.copy()
        if split == "coralscapes_test":
            # Select a reproducible high-fish-density view using only the ground truth.
            ref = np.asarray(reference_image) == 2
            h, w = ref.shape
            cw, ch = max(1, w//4), max(1, h//4)
            candidates = [(x, y, x+cw, y+ch) for y in np.linspace(0, h-ch, 7, dtype=int)
                          for x in np.linspace(0, w-cw, 7, dtype=int)]
            crop = max(candidates, key=lambda box: ref[box[1]:box[3], box[0]:box[2]].sum())
            reference_image = reference_image.crop(crop)
        with Image.open(row["image"]) as src:
            rgb = src.convert("RGB")
            if crop:
                rgb = rgb.crop(crop)
            rgb.thumbnail((460, 300), Image.Resampling.LANCZOS)
        reference = np.array(reference_image.resize(rgb.size, Image.Resampling.NEAREST))
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(SPLIT_NAMES[split], fontsize=9)
        axes[r, 1].imshow(overlay(rgb, reference, split != "uiis_test"))
        for j, arm in enumerate(ARMS, 2):
            file = out / "predictions" / arm / f"seed-{SEEDS[0]}" / split / f"{index:05d}.png"
            with Image.open(file) as src:
                region = src.crop(crop) if crop else src
                ids = np.array(region.resize(rgb.size, Image.Resampling.NEAREST))
            painted = overlay(rgb, ids, split != "uiis_test")
            painted[reference == 255] = np.array(overlay(rgb, reference, split != "uiis_test"))[reference == 255]
            axes[r, j].imshow(painted)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        metadata.append(dict(dataset=split, image_id=row["id"], group=row["group"], fixed_seed=SEEDS[0],
                             crop_xyxy=",".join(map(str,crop)) if crop else "full"))
    legend = [Patch(facecolor=PLOT_PALETTE[i] / 255, label=LABELS[i]) for i in range(1, len(LABELS))]
    legend.append(Patch(facecolor="0.65", label="ignored"))
    fig.legend(handles=legend, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(out / "figures" / "prediction-examples.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig)
    write_csv(out / "metrics" / "displayed-examples.csv", metadata)


def display(v, places=1, signed=False):
    if not np.isfinite(v):
        return "—"
    if signed and 0 < abs(v) < 0.5 * 10**(-places):
        places = 4
    return format(v, ("+" if signed else "") + f".{places}f").replace("-", "−")


def table(headers, body):
    heads = "".join("<th>" + html.escape(str(h)) + "</th>" for h in headers)
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in row) + "</tr>" for row in body)
    return f"<div class='scroll'><table><thead><tr>{heads}</tr></thead><tbody>{rows}</tbody></table></div>"


def embedded(path):
    typ = "jpeg" if path.suffix == ".jpg" else "png"
    return "data:image/" + typ + ";base64," + base64.b64encode(path.read_bytes()).decode()


def exposure_results(out, main, mean):
    def read_metrics(name):
        with (out/"metrics"/name).open(newline="") as file:
            return list(csv.DictReader(file))

    audit = json.loads((out/"reproduction/exposure/generator-overlap-summary.json").read_text())
    sens = read_metrics("uiis-overlap-sensitivity.csv")
    diff = read_metrics("uiis-overlap-paired-deltas.csv")
    idx = {(row["subset"], row["arm"], row["metric"]): row for row in sens}
    subset_titles = {"no_UIIS_train_match":"No original UIIS-train match", "no_known_generator_source_match":"No screened generator-source match"}
    def number(value):
        return np.nan if value in ("", None) else float(value)
    def pct(value):
        return display(number(value),2)
    def interval(r):
        lo = number(r["lower95_pp"])
        hi = number(r["upper95_pp"])
        return display(lo,2,True)+" to "+display(hi,2,True)
    def pct_diff(value):
        return display(number(value),2,True)
    result_rows = [["Full locked holdout", str(691)] +
                   [display(100*mean[("uiis_test", a, "foreground_mIoU")],2) for a in ARMS] +
                   [display(main["difference_pp"],2,True), interval(main)]]
    for key, title in subset_titles.items():
        pair = next(r for r in diff if r["subset"]==key and r["arm"]=="mixed" and
                    r["reference_arm"]=="real" and r["metric"]=="foreground_mIoU")
        result_rows.append([title, pair["images"]] +
                           [pct(idx[key,a,"foreground_mIoU"]["seed_mean_percent"]) for a in ARMS] +
                           [pct_diff(pair["delta_pp"]), interval(pair)])
    kept = idx["no_known_generator_source_match","real","foreground_mIoU"]["images"]
    robots = idx["no_known_generator_source_match","real","robots_IoU"]["reference_images_with_class"]
    wrecks = idx["no_known_generator_source_match","real","wrecks_IoU"]["reference_images_with_class"]
    full_pair = next(r for r in diff if r["subset"]=="no_known_generator_source_match" and
                     r["metric"]=="foreground_mIoU" and r["arm"]=="mixed" and r["reference_arm"]=="real")
    flagged = audit["flagged_images"]
    seen = sum(audit["photos_matched_to_names_by_source"].values())
    with (out/"reproduction/exposure/generator-image-matches.csv").open(newline="") as file:
        exact = len({r["id"] for r in csv.DictReader(file) if r["split"] == "uiis_test" and r["byte_identical"] == "1"})
    direct = audit["directly_matched_images"]["uiis_test"]
    return (
        "<h2>Potential generator-training overlap</h2><p>The study screened "
        f"{seen:,} of {audit['generator_index_photos']:,} images listed for TIDE generator training. "
        f"The screen flagged <strong>{flagged['uiis_test']} of 691 UIIS holdout frames</strong>. "
        f"{direct} matched a generator image directly, including {exact} with identical file bytes. "
        "Both visual hashes had to agree within four bits. Strongly transformed images can escape detection. "
        "Twelve inspected pairs depict the same views. "
        "<a href='reproduction/exposure/matched-photo-examples.jpg'>See the matched photographs.</a></p>" +
        table(["UIIS cohort", "Frames", "Synthetic mIoU", "50/50 mIoU", "Real mIoU", "50/50 − real", "95% interval"],result_rows) +
        f"<p>Among the {kept} frames left after screening all available sources, {robots} contain annotated robots "
        f"and {wrecks} contain wrecks. The filtered six-class interval uses "
        f"{int(full_pair['draws_with_required_reference_classes']):,} of {N_BOOT:,} bootstrap draws with all six classes "
        "represented. Sparse class coverage limits interpretation. The same source screen flagged "
        f"{flagged['deepfish_test']} DeepFish and {flagged['coralscapes_test']} CoralscapesV2 test frames. "
        "The external fish scores therefore provide a complementary measure of transfer.</p>"
    )


def make_report(out, manifests, rows, lookup, data, habitat):
    heading = "SegFormer on synthetic and real underwater images"
    mean = {(r["dataset"],r["arm"],r["metric"]): r["seed_mean"] for r in rows["summary"]}
    contrasts = rows["differences"]
    main = next(r for r in contrasts if r["dataset"] == "uiis_test" and r["metric"] == "foreground_mIoU"
                and r["arm"] == "mixed" and r["reference_arm"] == "real")
    delta = main["difference_pp"]
    sentence = (f"Replacing half the real training masks with SynTIDE "
                f"{'raised' if delta >= 0 else 'lowered'} six-class UIIS mean IoU by "
                f"<strong>{abs(delta):.2f} percentage points</strong> "
                f"(paired 95% interval {display(main['lower95_pp'],2,True)} to {display(main['upper95_pp'],2,True)}).")
    if main["lower95_pp"] <= 0 <= main["upper95_pp"]:
        sentence += " The interval includes both a loss and a gain on this benchmark."
    body = ["<h1>" + heading + "</h1><p class='lead'>" + sentence + "</p>"]
    ext_deltas = [100 * (mean[(s,"mixed","fish_IoU")] - mean[(s,"real","fish_IoU")]) for s in DATASETS[1:]]
    body.append(f"<p>With the same replacement, fish IoU changed by {display(ext_deltas[0],2,True)} points "
                f"on DeepFish and {display(ext_deltas[1],2,True)} points on CoralscapesV2.</p>")
    exposure = json.loads((out/"reproduction/exposure/generator-overlap-summary.json").read_text())
    overlaps = exposure["flagged_images"]["uiis_test"]
    body.append(f"<p><strong>Interpret the full UIIS result as an in-domain comparison.</strong> "
                f"{overlaps} of its 691 holdout frames match photographs from the generator’s training "
                f"sources. The sensitivity analysis below repeats the comparison on the remaining "
                f"{691 - overlaps} frames. Both external fish tests passed the same similarity screen.</p>")
    body.append("<h2>Segmentation accuracy</h2><p>Scores are percentages. Each entry is the arithmetic mean "
                "over three paired seeds. The dots show individual seed scores.</p>")
    body.append(table(["Training data", "UIIS foreground mIoU", "DeepFish fish IoU", "CoralscapesV2 fish IoU"],
                      [[ARM_NAMES[a]] + [display(100 * mean[(s,a,"foreground_mIoU" if s == "uiis_test" else "fish_IoU")],2)
                        for s in DATASETS] for a in ARMS]))
    body.append("<img alt='Per-seed IoU on each holdout' src='" + embedded(out/"figures"/"training-comparison.png") + "'>")
    primary_contrasts = [r for r in contrasts if r["metric"] == ("foreground_mIoU" if r["dataset"] == "uiis_test" else "fish_IoU")]
    body.append(exposure_results(out, main, mean))
    body.append("<h2>Paired differences</h2><p>Each difference subtracts the comparator’s score from "
                "the first condition’s score, in percentage points.</p>")
    body.append("<img alt='50/50 minus real-only score differences with paired 95% intervals and a zero reference' "
                "src='" + embedded(out/"figures"/"replacement-differences.png") + "'>")
    body.append(table(["Test set", "Training data", "Comparator", "Difference", "Paired 95% interval"],
                      [[SPLIT_NAMES[r["dataset"]], ARM_NAMES[r["arm"]], ARM_NAMES[r["reference_arm"]],
                        display(r["difference_pp"],2,True),
                        display(r["lower95_pp"],2,True) + " to " + display(r["upper95_pp"],2,True)]
                       for r in primary_contrasts]))
    h = next(r for r in habitat if r["arm"] == "mixed" and r["reference_arm"] == "real")
    clip = next(r for r in primary_contrasts if r["dataset"] == "deepfish_test" and r["arm"] == "mixed"
                and r["reference_arm"] == "real")
    body.append(f"<p>The {clip['bootstrap_groups']} DeepFish clips come from {h['habitats']} labeled habitats. "
                "Resampling whole habitats as a sensitivity check gives a mixed-minus-real fish IoU interval "
                f"of {display(h['lower95_pp'],2,True)} to {display(h['upper95_pp'],2,True)} points, "
                f"compared with {display(clip['lower95_pp'],2,True)} to {display(clip['upper95_pp'],2,True)} "
                "when resampling clips. The point estimate is shared by both analyses. "
                "<a href='metrics/deepfish-habitat-sensitivity.csv'>Habitat-level intervals for every comparison</a></p>")
    by_class = {(r["class_id"],r["arm"]): r for r in rows["class_iou"]}
    body.append("<h2>UIIS class results</h2><p>Pixels and image counts refer to the human-labeled test set. "
                "The fish, reef, and wreck labels also cover other vertebrates, invertebrates, and ruins, respectively. "
                "Each class contributes equal weight to foreground mIoU. Per-class Dice, precision and recall "
                "are included in <a href='metrics/class_iou.csv'>the class metrics CSV</a>.</p>")
    body.append(table(["Class", "Images", "Labeled pixels", "Synthetic IoU", "50/50 IoU", "Real IoU"],
                      [[name, by_class[i,"real"]["images_with_reference"], f"{by_class[i,'real']['reference_pixels']:,}"]
                       + [display(100*by_class[i,a]["seed_mean_iou"],2) for a in ARMS] for i,name in enumerate(LABELS) if i]))
    body.append(table(["Training data", "Foreground macro Dice", "Foreground macro precision", "Foreground macro recall"],
                      [[ARM_NAMES[a]] + [display(100*mean[("uiis_test",a,"foreground_macro_"+m)],2)
                                         for m in ("dice","precision","recall")] for a in ARMS]))
    changes = [100*(by_class[i,"mixed"]["seed_mean_iou"] - by_class[i,"real"]["seed_mean_iou"]) for i in range(1,7)]
    high,low=int(np.nanargmax(changes)),int(np.nanargmin(changes))
    body.append(f"<p>The largest mixed-versus-real difference was {display(changes[high],2,True)} points for "
                f"{LABELS[high+1]}. The smallest was {display(changes[low],2,True)} for {LABELS[low+1]}.</p>")
    body.append("<h2>Fish transfer</h2><p>Each label map is reduced to fish versus everything else after "
                "the model predicts all seven classes. False fish area is the fraction of valid pixels predicted "
                "as fish across frames containing zero annotated fish. A dash denotes zero denominator.</p>")
    metrics=["fish_IoU","fish_Dice","fish_precision","fish_recall","negative_fish_area"]
    body.append(table(["Test set","Training data"] + [METRIC_NAMES[m] for m in metrics],
                      [[SPLIT_NAMES[s], ARM_NAMES[a]] + [display(100*mean[(s,a,m)],2) for m in metrics]
                       for s in DATASETS[1:] for a in ARMS]))
    site = {(r["group"],r["arm"]):r for r in rows["groups"] if r["dataset"] == "coralscapes_test"}
    reference_stats = []
    for split in DATASETS[1:]:
        cm = data[split][0][0, 0]
        fish_pixels = cm[:, 2, :].sum()
        empty = (cm[:, 2, :].sum(axis=-1) == 0).sum()
        reference_stats.append(f"{fish_pixels / cm.sum() * 100:.2f}% of valid pixels on "
                               f"{SPLIT_NAMES[split]} ({int(empty)} fish-free frames)")
    body.append("<p>Annotated fish occupy " + " and ".join(reference_stats) +
                ". A low foreground fraction makes precision and false fish area sensitive to relatively "
                "small amounts of background oversegmentation.</p>")
    size = json.loads((out / "metrics" / "coralscapes-fish-size.json").read_text())
    body.append(f"<p>CoralscapesV2’s official instance masks identify {size['labeled_frame_instances']:,} "
                f"fish occurrences across the {size['frames']} test frames. They agree pixel for pixel with "
                f"the semantic fish masks. At the model’s {size['side']}-pixel input size, the median fish "
                f"occupies {size['median_input_equivalent_pixel_area']:.1f} input-equivalent pixels, and "
                f"{100*size['fraction_under_16_input_pixels']:.1f}% occupy less than 16. These area estimates "
                "rescale each native mask by the fitted image area ratio. This provides context for recognizing "
                "fish at this resolution. The effect of input size was not tested.</p>")
    body.append("<h3>CoralscapesV2 sites</h3>" + table(["Dive site","Frames","Synthetic fish IoU","50/50 fish IoU","Real fish IoU"],
                [[g,site[g,"real"]["frames"]] + [display(100*site[g,a]["seed_mean_fish_iou"],2) for a in ARMS]
                 for g in sorted({group for group, arm in site})]))
    body.append("<h2>Example predictions</h2><p>All examples use seed 17 and were selected by a fixed image-ID "
                "hash within ground-truth strata. The external rows show fish only. To make small fish visible, "
                "the CoralscapesV2 rows show crops one quarter of the frame width and height. A fixed grid "
                "selects the crop with the most reference fish pixels. Gray marks excluded pixels.</p>")
    body.append("<img alt='RGB images with reference and predictions for three runs' src='" +
                embedded(out/"figures"/"prediction-examples.jpg") + "'>")
    side = json.loads((out/"metrics"/"real"/"seed-17"/"uiis_test"/"complete.json").read_text())["input_side"]
    train_sizes = {a: len(read_manifest(manifests/f"{a}.csv")) for a in ARMS}
    mixed = read_manifest(manifests/"mixed.csv")
    counts = {s:sum(r["source"]==s for r in mixed) for s in set(r["source"] for r in mixed)}
    body.append(f"<h2>Design and interpretation</h2><p>Each SegFormer-B0 run started with an ImageNet-pretrained "
                f"encoder and a fresh semantic head. Real-only training used {train_sizes['real']:,} real pairs. "
                f"Synthetic-only training used {train_sizes['synthetic']:,} generated pairs. Mixed training used " +
                " and ".join(f"{n:,} {html.escape(s)} pairs" for s,n in sorted(counts.items())) +
                f". Each mixed batch comprised four real and four synthetic images, drawn from subsets of the "
                f"corresponding single-source training pools. All nine runs used 4,000 AdamW updates of eight "
                f"images each and the same flip and brightness augmentations. Peak learning rates were "
                f"0.00006 for the encoder and 0.0006 for the decoder, with 100 warm-up steps and linear "
                f"decay. The final checkpoint was scored for each paired seed (17, 29 and 43). The input "
                f"was fitted and padded to {side} × {side}. Evaluation removed padding, restored logits to the "
                f"original mask dimensions with bilinear interpolation, and selected the highest-scoring class "
                f"per pixel. The run configurations contain the exact schedule.</p>")
    body.append("<p>The 50/50 split is defined by image count. Over one pass through its 3,000 distinct "
                "images, synthetic examples supply 56.52% of valid supervised input pixels. At the "
                "512-pixel input size, all pixels on the synthetic canvases are supervised, compared "
                "with 76.93% on the real canvases after letterboxing and removal of ignored regions.</p>")
    real_rows = read_manifest(manifests/"real.csv")
    mixed_real = [r for r in mixed if r["source"] == "uiis"]
    real_groups = len({r["group"] or r["id"] for r in real_rows})
    mixed_real_groups = len({r["group"] or r["id"] for r in mixed_real})
    body.append(f"<p>The real-only pool contains {real_groups:,} UIIS scene or perceptual-hash groups "
                f"among {len(real_rows):,} distinct files. The mixed pool contains {mixed_real_groups:,} such "
                f"groups among {len(mixed_real):,} real files. Distinct views within the same group can be similar. "
                f"Matching groups across development and evaluation were excluded from training.</p>")
    real_robot_count = sum("6" in r["classes"].split("|") for r in real_rows)
    synthetic_rows = read_manifest(manifests/"synthetic.csv")
    synth_robot_count = sum("6" in r["classes"].split("|") for r in synthetic_rows)
    body.append(f"<p>The real training pool has {real_robot_count} images containing labeled robot pixels. "
                f"The synthetic pool has {synth_robot_count}. Differences reflect the observed source distributions, "
                f"including class frequency, label quality and scene diversity.</p>")
    body.append("<p>IoU pools pixel confusion counts across images before division. Dice, precision and recall "
                "use the same pooled fish counts. Background and ignore pixels are distinct. UIIS background is "
                "included when counting false positives and excluded from the six-class mean. Ignore pixels are "
                "excluded from every metric. Blank per-image scores indicate undefined denominators.</p>")
    body.append("<p>Intervals use 2,000 paired percentile bootstrap draws. Each draw resamples identical test "
                "groups for every condition and seed, pools pixel counts within each run, then averages the three "
                "paired seed differences. Draws missing a reference class are omitted for UIIS foreground mIoU. "
                "UIIS groups use available near-duplicate clusters or individual images. DeepFish groups are "
                "clip prefixes inferred from filenames. CoralscapesV2 groups are dive sites. Its five sites limit "
                "the precision of its interval. Seed standard deviations are reported separately in the CSVs. "
                "These intervals describe test-set sampling variability conditional on the trained models and "
                "selected training pools.</p>")
    body.append("<p>The TIDE generator was trained on the UIIS training split. The 691-image UIIS holdout "
                "was reserved in the TIDE paper and is evaluated here after training. The published UIIS "
                "fish category also includes other vertebrates, whereas the external sets target fish. "
                "Dataset changes can therefore reflect both scene differences and labeling policy. "
                "The comparison estimates the effect of replacing selected real labels under a fixed total budget. "
                "A benefit from adding synthetic data to a fixed set of real labels would require a separate "
                "half-size real-only control.</p>")
    body.append("<p>Manual inspection of selected SynTIDE overlays found an example with part of a sea star "
                "labeled as robot (TIDE_011743). This illustrates a possible source of error. Its frequency "
                "was not estimated by that inspection.</p>")
    body.append("<h2>Result files</h2><p>Seed scores, class support, per-site scores and bootstrap intervals "
                "are saved in <code>metrics/</code>. Each run’s per-image CSV maps the original identifier to "
                "its native-resolution indexed PNG in <code>predictions/</code>. Pixel IDs follow the study "
                "class map. Each <code>confusions.npz</code> stores one 7 × 7 ground-truth-by-prediction matrix "
                "per frame plus fish-free counts. Model checkpoints are in <code>runs/</code>.</p>")
    body.append("<p>Sources &nbsp; <a href='https://arxiv.org/html/2503.21771v2'>TIDE paper</a> · "
                "<a href='https://huggingface.co/datasets/LiamLian0727/UIIS'>UIIS</a> · "
                "<a href='https://alzayats.github.io/DeepFish/'>DeepFish</a> · "
                "<a href='https://huggingface.co/datasets/josauder/coralscapesV2'>CoralscapesV2</a> · "
                "<a href='https://huggingface.co/nvidia/mit-b0'>SegFormer-B0 encoder</a></p>")
    style = ("body{font:16px/1.55 system-ui,sans-serif;color:#192e3b;max-width:1040px;margin:0 auto;padding:36px 22px}"
             "h1{line-height:1.12;font-size:clamp(30px,5vw,49px)}h2{margin:36px 0 6px;font-size:25px}"
             "p{max-width:95ch}.lead{font-size:21px}img{width:100%;height:auto;margin:14px 0}"
             "table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px 10px;"
             "border-bottom:1px solid #cdd8de;font-variant-numeric:tabular-nums}th{background:#e8f0f2}"
             ".scroll{overflow-x:auto}a{color:#00695f}code{background:#eef3f5;padding:1px 4px}")
    (out/"report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>"+heading+"</title><style>"+style+"</style><main>"
        +"\n".join(body)+"</main></html>", encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/segformer-study"))
    parser.add_argument("--manifests", type=Path, default=Path("work/segstudy/data/manifests"))
    args=parser.parse_args()
    out=args.output
    (out/"figures").mkdir(parents=True,exist_ok=True)
    rows={k:[] for k in ["per_run","summary","class_iou","differences","groups"]}
    data={s:load_split(out,s) for s in DATASETS}
    lookup={s:summarize(out,s,*data[s],rows) for s in DATASETS}
    for name,values in rows.items():
        write_csv(out/"metrics"/f"{name}.csv", values)
    effect_plot(out,lookup)
    replacement_plot(out,rows["differences"])
    draw_examples(out,args.manifests,data)
    habitat = deepfish_habitat_differences(out, data["deepfish_test"])
    make_report(out,args.manifests,rows,lookup,data,habitat)
    print(f"Saved {out/'report.html'}", flush=True)


if __name__=="__main__":
    main()
