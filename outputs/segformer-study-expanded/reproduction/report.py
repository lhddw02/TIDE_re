"""Build the four-condition follow-up from the original and expanded study outputs."""
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ARMS = ("synthetic", "mixed", "real", "combined")
SEEDS = (17, 29, 43)
NAMES = {"synthetic": "Synthetic only", "mixed": "Original mix", "real": "Real only", "combined": "Expanded mix"}
BUDGETS = {"synthetic": (3000, 0), "mixed": (1500, 1500), "real": (0, 3000), "combined": (3000, 3000)}
SPLITS = ("uiis_test", "deepfish_test", "coralscapes_test")
SCREEN = "uiis_without_generator_match"
UIIS_SCREEN = "uiis_without_uiis_train_match"
HABITAT = "deepfish_by_habitat"
HEADLINE = ("uiis_test", SCREEN, "deepfish_test", "coralscapes_test")
DATASET_NAMES = {"uiis_test":"UIIS", SCREEN:"Screened UIIS", UIIS_SCREEN:"UIIS-train screen",
                 "deepfish_test":"DeepFish", "coralscapes_test":"CoralscapesV2", HABITAT:"DeepFish by habitat"}
CLASS = ("background", "wrecks", "fish", "reefs", "aquatic plants", "divers", "robots")
FISH = ("fish_IoU", "fish_Dice", "fish_precision", "fish_recall", "negative_fish_area")
PAIRS = (("combined", "real"), ("combined", "mixed"), ("combined", "synthetic"),
         ("mixed", "real"), ("synthetic", "real"), ("mixed", "synthetic"))
N_BOOT = 2000
PALETTE = np.array([[0,0,0],[83,133,255],[20,235,138],[255,184,60],[235,90,218],[255,85,70],[100,224,255]])


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(float(v), 10) if np.isfinite(v) else "")
                             if isinstance(v, (np.floating, float)) else v for k,v in row.items()})


def study(arm, previous, output):
    return output if arm == "combined" else previous


def load_dataset(previous, output, split):
    matrices, negatives = [], []
    ids = groups = None
    for arm in ARMS:
        arm_cm, arm_neg = [], []
        for seed in SEEDS:
            root = study(arm, previous, output) / "metrics" / arm / f"seed-{seed}" / split
            if not (root / "complete.json").is_file():
                raise FileNotFoundError(f"Evaluation incomplete in {root}")
            with np.load(root / "confusions.npz", allow_pickle=False) as source:
                if ids is None:
                    ids, groups = source["id"], source["group"]
                if not (np.array_equal(source["id"], ids) and np.array_equal(source["group"], groups)):
                    raise ValueError(f"Image identifiers or groups disagree in {root}")
                arm_cm.append(source["cm"])
                arm_neg.append(source["negative"])
        matrices.append(arm_cm)
        negatives.append(arm_neg)
    matrices = np.asarray(matrices)
    reference_pixels = matrices.sum(axis=-1)
    if not np.all(reference_pixels == reference_pixels[0, 0]):
        raise ValueError(f"Unpaired reference pixels in {split}")
    return matrices, np.asarray(negatives), ids, groups


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def stats(cm, neg, semantic):
    reference, prediction = cm.sum(-1), cm.sum(-2)
    hits = cm.diagonal(axis1=-2, axis2=-1)
    all_classes = {
        "IoU":divide(hits, reference + prediction - hits),
        "Dice":divide(2 * hits, reference + prediction),
        "precision":divide(hits, prediction), "recall":divide(hits, reference),
    }
    metrics = {f"fish_{m}":a[..., 2] for m, a in all_classes.items()}
    metrics["negative_fish_area"] = divide(neg[..., 0], neg[..., 1])
    if semantic:
        for m, a in all_classes.items():
            label = "foreground_mIoU" if m == "IoU" else "foreground_macro_" + m.lower()
            fg = a[..., 1:]
            metrics[label] = divide(np.nansum(fg, axis=-1), np.isfinite(fg).sum(axis=-1))
        for i, label in enumerate(CLASS[1:], 1):
            metrics[f"class_{i}_IoU"] = all_classes["IoU"][..., i]
    return metrics, all_classes


def primary(cohort):
    return "foreground_mIoU" if cohort.startswith("uiis") else "fish_IoU"


def summarize(cohort, cm, neg, groups, rng_seed, rows, include_scores=True):
    semantic = cohort.startswith("uiis")
    point, per_class = stats(cm.sum(2), neg.sum(2), semantic)
    group_ids, inverse = np.unique(groups, return_inverse=True)
    group_cm = np.stack([cm[:, :, inverse == g].sum(2) for g in range(len(group_ids))])
    group_neg = np.stack([neg[:, :, inverse == g].sum(2) for g in range(len(group_ids))])
    support = cm[0, 0].sum(axis=-1)
    for arm_index, arm in enumerate(ARMS):
        if not include_scores:
            continue
        for metric, values in point.items():
            v = values[arm_index]
            rows["summary"].append(dict(dataset=cohort, arm=arm, metric=metric, images=len(groups),
                                        groups=len(group_ids), seed_mean=v.mean(), seed_sd=v.std(ddof=1)))
            for seed, value in zip(SEEDS, v):
                rows["per_run"].append(dict(dataset=cohort, arm=arm, seed=seed, metric=metric, fraction=value))
        if semantic:
            for j, label in enumerate(CLASS):
                record = dict(dataset=cohort, arm=arm, class_id=j, class_name=label,
                              images_with_reference=int((support[:, j] > 0).sum()), reference_pixels=int(support[:, j].sum()))
                for metric, values in per_class.items():
                    v = values[arm_index, :, j]
                    record[f"seed_mean_{metric.lower()}"] = v.mean()
                    record[f"seed_sd_{metric.lower()}"] = v.std(ddof=1)
                rows["class_iou"].append(record)
        if cohort in SPLITS[1:]:
            group_metrics, _ = stats(group_cm, group_neg, semantic=False)
            for i, group in enumerate(group_ids):
                rows["groups"].append(dict(dataset=cohort, group=group, arm=arm,
                                           images=int((inverse == i).sum()), seed_mean_fish_iou=group_metrics["fish_IoU"][i,arm_index].mean()))

    weights = np.random.default_rng(rng_seed).multinomial(len(group_ids), [1. / len(group_ids)] * len(group_ids),
                                                          size=N_BOOT).astype(float)
    def resample(arr):
        return (weights @ arr.reshape(len(group_ids), -1)).reshape((N_BOOT,) + arr.shape[1:])
    boot_cm = resample(group_cm)
    boot, _ = stats(boot_cm, resample(group_neg), semantic)
    complete = (boot_cm[:, 0, 0, 1:, :].sum(axis=-1) > 0).all(axis=-1)
    for arm, comparator in PAIRS:
        a, b = ARMS.index(arm), ARMS.index(comparator)
        for metric, values in point.items():
            if not include_scores and metric != "fish_IoU":
                continue
            paired = 100 * (values[a] - values[b])
            boot_delta = 100 * (boot[metric][:, a] - boot[metric][:, b]).mean(axis=-1)
            if metric.startswith("foreground_"):
                boot_delta = boot_delta[complete]
            elif metric.startswith("class_"):
                class_id = int(metric.split("_")[1])
                boot_delta = boot_delta[boot_cm[:, 0, 0, class_id, :].sum(axis=-1) > 0]
            boot_delta = boot_delta[np.isfinite(boot_delta)]
            lo, hi = np.quantile(boot_delta, [.025, .975]) if len(boot_delta) else (np.nan, np.nan)
            rows["differences"].append(dict(dataset=cohort, arm=arm, reference_arm=comparator, metric=metric,
                                            difference_pp=paired.mean(), lower95_pp=lo, upper95_pp=hi,
                                            paired_seed_sd_pp=paired.std(ddof=1), bootstrap_groups=len(group_ids),
                                            valid_bootstraps=len(boot_delta),
                                            **{f"seed_{seed}_difference_pp": d for seed, d in zip(SEEDS, paired)}))


def summarize_all(previous, output):
    rows = {k: [] for k in ("summary", "per_run", "class_iou", "differences", "groups")}
    flags = {row["id"]:row for row in read_rows(previous / "reproduction/exposure/generator-overlap-flags.csv")
             if row["split"] == "uiis_test"}
    habitats = {row["id"]:row["habitat"] for row in read_rows(previous / "reproduction/manifests/deepfish-habitats.csv")}
    for i, split in enumerate(SPLITS):
        cm, neg, ids, groups = load_dataset(previous, output, split)
        summarize(split, cm, neg, groups, 781 + i, rows)
        if split == "uiis_test":
            for cohort, criterion in ((SCREEN, "known_overlap"), (UIIS_SCREEN, "matched_uiis")):
                selected = np.array([flags[name][criterion] == "0" for name in ids])
                summarize(cohort, cm[:, :, selected], neg[:, :, selected], groups[selected], 1082, rows)
        if split == "deepfish_test":
            summarize(HABITAT, cm, neg, np.array([habitats[name] for name in ids]), 1832, rows, include_scores=False)
    return rows


def headline_table(rows):
    summaries = {(r["dataset"],r["arm"],r["metric"]):r for r in rows["summary"]}
    records = []
    for arm in ARMS:
        synthetic, real = BUDGETS[arm]
        record = dict(arm=arm, training_condition=NAMES[arm], synthetic_images=synthetic, real_images=real,
                      unique_training_images=synthetic+real, optimizer_updates_per_seed=4000)
        for cohort in HEADLINE:
            metric = primary(cohort)
            result = summaries[(cohort,arm,metric)]
            record[f"{cohort}_{metric}_percent"] = 100*result["seed_mean"]
            record[f"{cohort}_{metric}_seed_sd_pp"] = 100*result["seed_sd"]
        records.append(record)
    return records


def check_original(previous, rows):
    """Require every published summary and paired interval to reproduce to rounding precision."""
    summary = {(r["dataset"],r["arm"],r["metric"]):r for r in rows["summary"]}
    diffs = {(r["dataset"],r["arm"],r["reference_arm"],r["metric"]):r for r in rows["differences"]}
    prior = previous / "metrics"
    for r in read_rows(prior / "summary.csv"):
        rec = summary[(r["dataset"],r["arm"],r["metric"])]
        for field in ("seed_mean", "seed_sd"):
            np.testing.assert_allclose(rec[field], float(r[field]), atol=1e-9, rtol=0)
    for r in read_rows(prior / "differences.csv"):
        rec = diffs[(r["dataset"],r["arm"],r["reference_arm"],r["metric"])]
        for field in ("difference_pp","lower95_pp","upper95_pp"):
            np.testing.assert_allclose(rec[field],float(r[field]),atol=1e-7,rtol=0)
    for r in read_rows(prior / "uiis-overlap-paired-deltas.csv"):
        metric = r["metric"]
        if metric != "foreground_mIoU":
            label = metric[:-len("_IoU")]
            metric = f"class_{CLASS.index(label)}_IoU"
        cohort = SCREEN if r["subset"] == "no_known_generator_source_match" else UIIS_SCREEN
        rec = diffs[(cohort,r["arm"],r["reference_arm"],metric)]
        for old, new in (("delta_pp","difference_pp"),("lower95_pp","lower95_pp"),("upper95_pp","upper95_pp")):
            np.testing.assert_allclose(float(r[old]), rec[new], atol=1e-6, rtol=0)
    for r in read_rows(prior / "deepfish-habitat-sensitivity.csv"):
        rec = diffs[(HABITAT,r["arm"],r["reference_arm"],"fish_IoU")]
        for field in ("difference_pp","lower95_pp","upper95_pp"):
            np.testing.assert_allclose(rec[field],float(r[field]),atol=1e-7,rtol=0)


def fmt(value, signed=False):
    if not np.isfinite(value):
        return "—"
    precision = 4 if signed and value != 0 and abs(value)<0.005 else 2
    spec = f"{'+' if signed else ''}.{precision}f"
    return format(value,spec).replace("-","−")


def interval(r):
    return f"{fmt(r['difference_pp'],True)} [{fmt(r['lower95_pp'],True)}, {fmt(r['upper95_pp'],True)}]"


def delta_index(rows):
    return {(r["dataset"], r["reference_arm"]):r for r in rows["differences"]
            if r["arm"] == "combined" and r["metric"] == primary(r["dataset"])}


def forest_figure(output, deltas):
    fig, ax = plt.subplots(figsize=(10,4.2))
    fig.subplots_adjust(left=.28,right=.77,top=.89,bottom=.17)
    ax.axvline(0,color="#63717b",lw=1)
    ax.text(1.035, 1.005, "Difference [95% interval]", transform=ax.transAxes, fontsize=9, color="#46575d")
    specs=[("real",.14,"#095b57","o","vs. real only"),("mixed",-.14,"#92442c","s","vs. original mix")]
    for i, cohort in enumerate(HEADLINE):
        y = len(HEADLINE)-1-i
        for comparator, offset, color, marker, _ in specs:
            rec = deltas[(cohort,comparator)]
            lo, hi = rec["lower95_pp"], rec["upper95_pp"]
            position = y+offset
            ax.plot([lo,hi],[position,position],color=color,lw=2)
            ax.plot([lo,hi],[position,position],"|",color=color,ms=8)
            ax.plot(rec["difference_pp"],position,marker,color=color,ms=6)
            ax.text(1.035,position,interval(rec),ha="left",va="center",color=color,fontsize=9,
                    transform=ax.get_yaxis_transform())
    handles=[Line2D([0],[0],color=color,marker=marker,label=title) for _,_,color,marker,title in specs]
    fig.legend(handles=handles,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.50,1.015))
    labels=[f"{DATASET_NAMES[c]}\n{'Foreground mIoU' if c.startswith('uiis') else 'Fish IoU'}" for c in HEADLINE]
    ax.set_yticks(list(range(3,-1,-1)),labels)
    ax.set_ylim(-.50,3.50)
    ax.tick_params(axis="y",length=0,pad=10)
    ax.spines[["left","top","right"]].set_visible(False)
    ax.grid(axis="x",alpha=.15)
    ax.set_xlabel("Expanded mix minus comparator (percentage points)")
    ax.set_axisbelow(True)
    fig.savefig(output/"figures/paired-differences.png",dpi=180,bbox_inches="tight",pad_inches=.13)
    plt.close(fig)


def colorize(rgb, labels, ignore, binary=False):
    labels = np.where(labels==2,2,0) if binary else labels
    image = np.array(rgb, dtype=float)
    valid = (labels > 0) & (labels < 7) & ~ignore
    image[valid] = image[valid]*.28 + PALETTE[labels[valid]]*.72
    image[ignore] = image[ignore]*.35 + 160*.65
    return image.astype(np.uint8)


def example_figure(previous, output):
    examples = read_rows(previous/"metrics/displayed-examples.csv")
    lookup = {}
    for split in SPLITS:
        references=read_rows(previous/"reproduction/manifests"/f"{split}.csv")
        for i, r in enumerate(references):
            lookup[(split,r["id"])] = i,r
    columns = ["Photograph","Ground truth"]+[NAMES[a] for a in ARMS]
    fig, axes=plt.subplots(len(examples),len(columns),figsize=(14,1.6*len(examples)),squeeze=False)
    for ax, name in zip(axes[0], columns):
        ax.set_title(name,fontsize=11)
    for j, ex in enumerate(examples):
        split=ex["dataset"]
        i, ref=lookup[(split,ex["image_id"])]
        box=None if ex["crop_xyxy"]=="full" else tuple(map(int,ex["crop_xyxy"].split(",")))
        def native_window(path, rgb=False):
            with Image.open(path) as source:
                image = source.convert("RGB") if rgb else source.copy()
            return image.crop(box) if box is not None else image
        def read_image(path, rgb=False, size=None):
            im = native_window(path, rgb)
            if size is None:
                im.thumbnail((450,246),Image.Resampling.LANCZOS)
            else:
                im=im.resize(size,Image.Resampling.NEAREST)
            return im
        image=read_image(ref["image"],rgb=True)
        gt=np.asarray(read_image(ref["mask"],size=image.size))
        ignore=gt==255
        native_gt = np.asarray(native_window(ref["mask"]))
        ex["reference_fish_pixels_in_view"] = int((native_gt == 2).sum())
        axes[j,0].imshow(image)
        axes[j,0].set_ylabel(DATASET_NAMES[split],fontsize=8)
        axes[j,1].imshow(colorize(image,gt,ignore,split!="uiis_test"))
        for ax, arm in zip(axes[j,2:],ARMS):
            path=study(arm,previous,output)/"predictions"/arm/"seed-17"/split/f"{i:05d}.png"
            native_pred = np.asarray(native_window(path))
            ex[f"{arm}_true_fish_pixels_in_view"] = int(((native_gt == 2) & (native_pred == 2)).sum())
            pred=np.asarray(read_image(path,size=image.size))
            ax.imshow(colorize(image,pred,ignore,split!="uiis_test"))
        for ax in axes[j]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
    legend=[Patch(facecolor=PALETTE[i]/255,label=CLASS[i]) for i in range(1,7)]+[Patch(facecolor="#a0a0a0",label="ignored")]
    fig.legend(handles=legend,loc="lower center",ncol=7,frameon=False,fontsize=8,bbox_to_anchor=(.5,.0005))
    fig.tight_layout(rect=(0,.035,1,1),pad=.45)
    fig.savefig(output/"figures/prediction-comparison.jpg",dpi=180,bbox_inches="tight",pil_kwargs={"quality":88})
    plt.close(fig)
    write_rows(output/"metrics/displayed-examples.csv",examples)


def table(caption, columns, data, highlight=None):
    header = "".join(f'<th scope="col">{html.escape(str(h))}</th>' for h in columns)
    rendered = []
    for i, row in enumerate(data):
        cells = [f'<th scope="row">{html.escape(str(row[0]))}</th>']
        cells += [f"<td>{html.escape(str(v))}</td>" for v in row[1:]]
        rendered.append(('<tr class="highlight">' if i == highlight else "<tr>") + "".join(cells) + "</tr>")
    return (f'<div class="scroller"><table><caption>{html.escape(caption)}</caption><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rendered)}</tbody></table></div>')


def embed(path, alt):
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    image = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:{mime};base64,{image}" alt="{html.escape(alt,quote=True)}">'


def ci_text(rec):
    return f"{fmt(rec['lower95_pp'],True)} to {fmt(rec['upper95_pp'],True)}"


def interval_summary(rec):
    if rec["lower95_pp"] > 0:
        return "The interval favors the expanded pool."
    if rec["upper95_pp"] < 0:
        return "The interval favors the smaller comparison pool."
    return "The interval includes both a decrease and an increase."


def html_report(previous, output, rows):
    summary = {(r["dataset"],r["arm"],r["metric"]):r for r in rows["summary"]}
    per_class = {(r["dataset"],r["arm"],r["class_name"]):r for r in rows["class_iou"]}
    delta = delta_index(rows)
    previous_link = os.path.relpath(previous, start=output)
    def link(path, label):
        return f'<a href="{html.escape(str(path),quote=True)}">{html.escape(label)}</a>'
    def percent(cohort, arm, metric=None):
        return fmt(100 * summary[(cohort,arm,metric or primary(cohort))]["seed_mean"])
    def points(rec):
        return f"<strong>{fmt(rec['difference_pp'],True)} percentage points</strong> (paired 95% interval {ci_text(rec)})"
    shift_mixed, shift_real = delta[("uiis_test","mixed")], delta[("uiis_test","real")]
    same_real_contrast = interval_summary(shift_real)
    if shift_real["upper95_pp"] < 0:
        same_real_contrast = "The interval favors real-only training."
    both_uncertain = (shift_real["lower95_pp"] <= 0 <= shift_real["upper95_pp"] and
                      shift_mixed["lower95_pp"] <= 0 <= shift_mixed["upper95_pp"])
    mix_uncertainty = "" if both_uncertain else interval_summary(shift_mixed)
    if both_uncertain:
        same_real_contrast = "Both UIIS intervals include zero, so the sampled test groups leave the direction of either change uncertain."
    score_table = table("Mean overlap on real test images (%)",
        ["Training condition", "Synthetic images", "Real images", "UIIS mIoU", "Screened UIIS mIoU", "DeepFish fish IoU", "CoralscapesV2 fish IoU"],
        [[NAMES[a]]+[f"{n:,}" for n in BUDGETS[a]]+[percent(c,a) for c in HEADLINE] for a in ARMS],highlight=3)
    comparison_table=table("Expanded mix minus each comparator, percentage points [95% interval]",
        ["Test set and score","Against original mix","Against real only","Against synthetic only"],
        [[DATASET_NAMES[c]+(" · six-class mIoU" if c.startswith("uiis") else " · fish IoU")]+
         [interval(delta[(c,a)]) for a in ("mixed","real","synthetic")] for c in HEADLINE])
    class_rows=[[name, per_class[("uiis_test","real",name)]["images_with_reference"]] +
                [fmt(100*per_class[("uiis_test",a,name)]["seed_mean_iou"]) for a in ARMS] for name in CLASS[1:]]
    classes=table("UIIS per-class IoU (%)",["Class","Images with class"]+[NAMES[a] for a in ARMS],class_rows)
    fish=table("Fish segmentation on external test sets (%)",
        ["Test set", "Training condition", "IoU","Dice","Precision","Recall","False fish area"],
        [[DATASET_NAMES[c],NAMES[a]]+[percent(c,a,m) for m in FISH] for c in SPLITS[1:] for a in ARMS])
    main_screened=delta[(SCREEN,"real")]
    direct=delta[(UIIS_SCREEN,"real")]
    audit=json.loads((previous/"reproduction/exposure/generator-overlap-summary.json").read_text())
    magnitudes=[100*(per_class[("uiis_test","combined",label)]["seed_mean_iou"]-
                      per_class[("uiis_test","real",label)]["seed_mean_iou"]) for label in CLASS[1:]]
    top, low = int(np.argmax(magnitudes)), int(np.argmin(magnitudes))
    sizes=json.loads((previous/"metrics/coralscapes-fish-size.json").read_text())
    source_match=audit["flagged_images"]["uiis_test"]
    baseline_files = [link(f"{previous_link}/report.html","Original three-condition report"),
                       link("baseline-snapshot.zip","Original results archive")]
    report_files = [link("metrics/headline.csv","Headline score table"),link("metrics/summary.csv","All metrics and seed SDs"),link("metrics/differences.csv","All paired differences"),
                    link("metrics/per_run.csv","Individual seed scores"),link("metrics/class_iou.csv","Class metrics and support"),
                    link("metrics/groups.csv","External clip and site scores"),link("figures/paired-differences.png","Difference figure"),
                    link("figures/prediction-comparison.jpg","Prediction examples")]
    files = " · ".join(report_files)
    coral_examples = [r for r in read_rows(output/"metrics/displayed-examples.csv") if r["dataset"]=="coralscapes_test"]
    fish_in_crops = sum(int(r["reference_fish_pixels_in_view"]) for r in coral_examples)
    coral_example_text = ""
    if fish_in_crops and all(int(r[f"{a}_true_fish_pixels_in_view"]) == 0 for r in coral_examples for a in ARMS):
        coral_example_text = f" All four models assigned the {fish_in_crops:,} reference fish pixels across the two displayed reef crops to other classes."
    portrait=embed(output/"figures/prediction-comparison.jpg","Six reference examples with predictions for all four models, evaluated at seed 17")
    chart=embed(output/"figures/paired-differences.png","Paired differences for the expanded mix compared with real only and the original mix. Exact estimates and intervals appear in the following table.")
    body = f"""
<header><h1>Underwater segmentation with 6,000 training images</h1>
<p class="subtitle">SegFormer-B0 trained on 3,000 SynTIDE image–mask pairs and 3,000 human-labeled UIIS pairs, compared with the three original training conditions.</p></header>
<section>
<h2>What changed with twice the data?</h2>
<p class="lead">The expanded mix reached <strong>{percent("uiis_test","combined")}%</strong> six-class mean IoU on the 691-image UIIS holdout. Doubling the original balanced mix changed this score by {points(shift_mixed)}. {mix_uncertainty}</p>
<p>Compared with training on the same 3,000 real images alone, the change was {points(shift_real)}. {same_real_contrast}</p>
{score_table}
<p class="note">Each score is the mean of three paired seeds. UIIS weights the six foreground classes equally. DeepFish (186 images) and CoralscapesV2 (473 images) measure fish overlap. Screened UIIS contains 113 of the full holdout’s images.</p>
</section>
<section>
<h2>Effects across test sets</h2>
<p>Doubling the mixed pool changed DeepFish fish IoU from {percent("deepfish_test","mixed")}% to {percent("deepfish_test","combined")}%, a difference of {points(delta["deepfish_test","mixed"])}. On CoralscapesV2, the expanded mix scored {percent("coralscapes_test","combined")}% compared with {percent("coralscapes_test","real")}% for real-only training. The figure shows paired differences at the same training budget. Zero marks equal performance.</p>
<figure>{chart}</figure>{comparison_table}
<p class="note">The paired bootstrap draws resample 649 UIIS visual groups, 49 DeepFish recording groups, or five CoralscapesV2 sites, with 2,000 draws per test set. Resampling the 19 DeepFish habitats gives a 95% interval of {ci_text(delta[HABITAT,"real"])} points for the expanded mix versus real only. Five sites give limited information about performance on new reefs.</p>
</section>
<section>
<h2>Generator-source overlap</h2>
<p>Of the 691 UIIS holdout images, {source_match} closely matched photographs listed in TIDE generator training, including {audit["byte_identical_images"]["uiis_test"]} identical files. Across all {audit["generator_index_photos"]:,} indexed source photos, the screen found zero matches in either external test. The 113 UIIS images without a detected match yield an expanded-minus-real difference of {points(main_screened)}. They include only one robot image and three wreck images. Of 2,000 draws, {main_screened["valid_bootstraps"]:,} contained all six reference classes.</p>
<p>The narrower screen against the original UIIS training source alone retained 342 images and yielded an expanded-minus-real difference of {points(direct)}. Downstream training excluded candidate photos in groups matching the holdouts. The generator-source screen is a separate check on upstream exposure and can miss strongly transformed copies. {link(f"{previous_link}/reproduction/exposure/matched-photo-examples.jpg","Examples of generator-source matches")}</p>
</section>
<section>
<h2>Performance by class</h2>
{classes}
<p>The largest change from real-only training was {fmt(magnitudes[top],True)} points for {CLASS[top+1]}. The smallest was {fmt(magnitudes[low],True)} for {CLASS[low+1]}. The UIIS labels for fish and reefs include other vertebrates and invertebrates, respectively.</p>
{fish}
<p class="note">False fish area is the percentage of valid pixels labeled fish on frames with zero reference fish. DeepFish has 90 such frames and CoralscapesV2 has 54. CoralscapesV2 contains {sizes["labeled_frame_instances"]:,} labeled fish occurrences, with a median area equivalent to {sizes["median_input_equivalent_pixel_area"]:.1f} pixels at this model’s input scale.</p>
</section>
<section>
<h2>Examples from all four models</h2>
<figure>{portrait}<figcaption>Seed 17, using the same images and crops as the original study. The ground-truth masks and four predictions are overlaid on each photograph. CoralscapesV2 rows magnify a quarter-width, quarter-height crop selected from the reference mask to show small fish.{coral_example_text} External overlays show fish only. Metrics use the entire frame at original resolution. {link("metrics/displayed-examples.csv","Image identifiers and crop coordinates")}</figcaption></figure>
</section>
<section>
<h2>What the comparisons measure</h2>
<p><strong>Expanded mix versus original mix.</strong> The expanded pool contains all 3,000 images used by the smaller mix plus 1,500 additional images from each source. Both conditions presented 16,000 real and 16,000 synthetic images per run, with four from each source per batch. The same 32,000 presentations cover 5.33 passes through the expanded pool and 10.67 passes through the smaller one. This measures the benefit of greater training diversity at fixed compute for these selected pools. An equal-epoch comparison would require additional updates.</p>
<p><strong>Expanded mix versus real only.</strong> Both conditions have exactly the same 3,000 human-labeled images. The expanded condition adds 3,000 synthetic pairs while reducing presentations of the real images from 32,000 to 16,000 per run. Its result reflects both changes. Over one complete pass through the expanded pool, synthetic images contribute 56.79% of supervised pixels after padding and ignored pixels are excluded. The real pool contains 2,225 inferred visual groups. Each synthetic image comes from a different caption.</p>
<p>Every model used the same ImageNet-pretrained MiT-B0 encoder, newly initialized seven-class segmentation head, seeds 17, 29 and 43, and 4,000 AdamW updates with batches of eight. Inputs were fit and padded to 512 × 512. The shared recipe used flips and brightness augmentation, a 100-step warm-up, and linear learning-rate decay. Final checkpoints were evaluated after cropping the padding and restoring logits to the original mask size before class assignment.</p>
<p>Metrics pool valid pixel counts over each test set before averaging run scores. Background contributes false positives to foreground classes. Label 255 is excluded. Bootstrap draws resample the same test groups for all arms and seeds. Intervals quantify test-group sampling uncertainty conditional on these fitted models and training pools. Seed variation is reported separately. UIIS six-class draws require all six reference classes.</p>
</section>
<footer>
<h2>Saved results</h2>
<p>{files}</p><p>{" · ".join(baseline_files)}</p>
<p>Native prediction masks, per-image confusion counts and final model weights for the three new runs are saved in the new study’s <code>predictions</code>, <code>metrics/combined</code> and <code>runs/combined</code> directories. The archived original contains the other nine checkpoints and their predictions.</p>
<p class="sources">Sources &nbsp; {link("https://arxiv.org/abs/2503.21771","TIDE")} · {link("https://huggingface.co/datasets/LiamLian0727/UIIS","UIIS")} · {link("https://alzayats.github.io/DeepFish/","DeepFish")} · {link("https://huggingface.co/datasets/josauder/coralscapesV2","CoralscapesV2")} · {link("https://huggingface.co/nvidia/mit-b0","SegFormer encoder")}</p>
</footer>"""
    style = """
:root{color-scheme:light;--ink:#192e35;--soft:#53656a;--rule:#d2dddd;--accent:#0c635e}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:16px/1.6 'Source Sans 3',system-ui,sans-serif}
main{width:min(100%,1120px);margin:auto;padding:40px clamp(16px,4vw,64px) 72px}
header{border-top:5px solid var(--accent);padding-top:28px;margin-bottom:36px}
h1,h2{font-family:'Source Sans 3',system-ui,sans-serif;letter-spacing:-.025em;line-height:1.16}
h1{font-size:clamp(34px,5vw,58px);margin:0;max-width:22ch}h2{font-size:clamp(24px,3vw,32px);margin:0 0 12px}
.subtitle,.lead{font-size:clamp(17px,2vw,21px)}.subtitle{color:var(--soft);max-width:69ch}section,footer{padding:25px 0;border-top:1px solid var(--rule)}
p{max-width:88ch;margin:.65em 0 1em}.note,figcaption,.sources{font-size:14px;color:var(--soft)}
.scroller{overflow-x:auto;margin:1.3em 0}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.4;font-variant-numeric:tabular-nums}
caption{font-weight:700;color:var(--ink);text-align:left;margin-bottom:.6em}th,td{padding:8px;border-bottom:1px solid var(--rule);text-align:right}
thead th{font-weight:650;background:#eaf2f1;vertical-align:bottom}th:first-child{text-align:left}tbody th{font-weight:550}
.highlight{background:#ecf6f3}.highlight th,.highlight td{font-weight:700}img{display:block;width:100%;height:auto}figure{margin:1.2em 0}figcaption{margin:.8em 0}
a{color:#005e59;text-underline-offset:.15em}code{font-size:.9em}footer p{max-width:100ch}
@media print{main{padding:0}h2,caption{break-after:avoid}figure,tr{break-inside:avoid}}
"""
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
           'content="width=device-width, initial-scale=1"><title>Expanded SynTIDE and UIIS segmentation study</title>'
           f'<style>{style}</style></head><body><main>{body}</main></body></html>')
    (output / "report.html").write_text(doc, encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("outputs/segformer-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/segformer-study-expanded"))
    args=parser.parse_args()
    if args.previous.resolve()==args.output.resolve():
        raise ValueError("Choose a separate output directory for the expanded study")
    rows=summarize_all(args.previous,args.output)
    check_original(args.previous,rows)
    for name, records in rows.items():
        write_rows(args.output/"metrics"/(name+".csv"),records)
    write_rows(args.output/"metrics/headline.csv",headline_table(rows))
    (args.output/"figures").mkdir(exist_ok=True)
    forest_figure(args.output,delta_index(rows))
    example_figure(args.previous,args.output)
    html_report(args.previous,args.output,rows)
    print(f"Reproduced original scores and intervals, saved {args.output/'report.html'}", flush=True)


if __name__=="__main__":
    main()
