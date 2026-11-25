"""Produce the B4, B0 and Table 3 comparison from verified holdout confusions."""
import argparse
import base64
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

STAGES = ("real", "synthetic", "finetuned")
STAGE_NAMES = {"real": "Real only", "synthetic": "SynTIDE pretraining", "finetuned": "SynTIDE → UIIS"}
PAPER = {"real":70.2, "synthetic":76.5, "finetuned":75.4}
B0_NAMES = {"real":"Real only", "synthetic":"SynTIDE only", "mixed":"1,500 + 1,500 joint mix", "combined":"3,000 + 3,000 joint mix"}
CLASSES = ("Background","Wrecks","Fish","Reefs","Aquatic plants","Divers","Robots")
GROUPS = {"full":"Official UIIS", "no_uiis_match":"No detected UIIS-train match", "no_generator_match":"No detected generator-source match"}
PAIRS = (("finetuned","real"),("finetuned","synthetic"),("synthetic","real"))
COLORS = {"real":"#4c6c8a","synthetic":"#b76f22","finetuned":"#08695e"}
TABLE3 = "https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_A_Unified_Image-Dense_Annotation_Generation_Model_for_Underwater_Scenes_CVPR_2025_paper.pdf#page=7"
AUTHOR = "https://github.com/HongkLin/TIDE/issues/7#issuecomment-3501735490"
SETTINGS = "https://github.com/HongkLin/TIDE/issues/4#issuecomment-2994279572"
MASKS = "https://github.com/HongkLin/TIDE/issues/9"
FREEMASK = "https://papers.neurips.cc/paper_files/paper/2023/file/3ba7560b4c3e66d760fbdd472cf4a5a9-Paper-Conference.pdf#page=5"
FILTER_CODE = "https://github.com/LiheYoung/FreeMask/blob/main/preprocess/filter.py#L62-L101"


def read(path):
    with path.open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def save(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0])
        writer.writeheader()
        for row in records:
            writer.writerow({k: (f"{v:.8f}" if np.isfinite(v) else "") if isinstance(v,(float,np.floating)) else v
                             for k,v in row.items()})


def stats(matrix):
    tp = matrix.diagonal(axis1=-2, axis2=-1)
    union = matrix.sum(axis=-2) + matrix.sum(axis=-1) - tp
    iou = np.divide(tp, union, out=np.full(tp.shape,np.nan), where=union > 0) * 100.
    return iou[...,1:].mean(axis=-1), iou


def load_arrays(study):
    mats, done = {}, {}
    ids = groups = None
    for stage in STAGES:
        directory = study / "metrics" / stage
        verified = json.loads((directory/"verified.json").read_text())
        meta = json.loads((directory/"complete.json").read_text())
        assert verified["verified_masks"] == meta["images"] == 691
        assert verified["checkpoint_sha256"] == meta["checkpoint_sha256"]
        assert abs(meta["foreground_miou_percent"] - verified["foreground_miou_percent"]) < 1e-8
        with np.load(directory/"confusions.npz", allow_pickle=False) as file:
            if ids is None:
                ids, groups = file["id"], file["group"]
            assert np.array_equal(ids, file["id"]) and np.array_equal(groups, file["group"])
            mats[stage] = file["cm"]
        assert np.array_equal(mats[stage].sum(axis=2), mats["real"].sum(axis=2))
        done[stage] = meta
    return mats, ids, groups, done


def selections(original, ids):
    flags = {r["id"]:r for r in read(original / "reproduction/exposure/generator-overlap-flags.csv") if r["split"]=="uiis_test"}
    assert set(ids) == set(flags)
    sel = {"full":np.ones(len(ids),dtype=bool)}
    for key, flag in (("no_uiis_match","matched_uiis"),("no_generator_match","known_overlap")):
        sel[key] = np.array([not int(flags[name][flag]) for name in ids])
    assert [int(v.sum()) for v in sel.values()] == [691,342,113]
    return sel


def summarize(mats, groups, sel):
    scores, classes, differences = [], [], []
    for c, keep in sel.items():
        names, inverse = np.unique(groups[keep],return_inverse=True)
        values = np.stack([mats[s][keep] for s in STAGES])
        reference = values[0].sum(axis=-1)
        present = (reference > 0).sum(axis=0)
        miou, iou = stats(values.sum(axis=1))
        for idx, stage in enumerate(STAGES):
            scores.append(dict(cohort=c, stage=stage, images=int(keep.sum()), visual_groups=len(names),
                               foreground_miou_percent=miou[idx], valid_pixels=int(reference.sum())))
            for label, name in enumerate(CLASSES):
                classes.append(dict(cohort=c, stage=stage, class_id=label, class_name=name,
                                    iou_percent=iou[idx,label], images_with_class=int(present[label]),
                                    reference_pixels=int(reference[:,label].sum())))
        grouped = np.stack([values[:,inverse==j].sum(axis=1) for j in range(len(names))])
        weights = np.random.default_rng(781 if c=="full" else 1082).multinomial(
            len(names), [1. / len(names)]*len(names), size=2000)
        resampled = (weights @ grouped.reshape(len(names),-1)).reshape((len(weights),)+grouped.shape[1:])
        averages, _ = stats(resampled)
        complete = (resampled[:,0,1:,:].sum(axis=-1)>0).all(axis=1)
        for first, second in PAIRS:
            a, b = STAGES.index(first), STAGES.index(second)
            draws = (averages[:,a]-averages[:,b])[complete]
            lo, hi = np.quantile(draws, [0.025,0.975])
            differences.append(dict(cohort=c, stage=first, comparator=second, difference_pp=miou[a]-miou[b],
                                    lower95_pp=lo, upper95_pp=hi, bootstrap_groups=len(names),
                                    bootstrap_draws=2000, draws_with_all_six_classes=len(draws)))
    return scores, classes, differences


def load_b0(original, expanded, ids, groups):
    b0 = []
    saved = {r["arm"]:r for r in read(expanded/"metrics/headline.csv")}
    for arm in ("real", "synthetic", "mixed", "combined"):
        root = expanded if arm=="combined" else original
        results = []
        for seed in (17,29,43):
            file = root / f"metrics/{arm}/seed-{seed}/uiis_test/confusions.npz"
            with np.load(file, allow_pickle=False) as data:
                assert np.array_equal(data["id"],ids) and np.array_equal(data["group"],groups)
                mean,_ = stats(data["cm"].sum(axis=0))
                results.append(float(mean))
        prior = saved[arm]
        assert abs(np.mean(results)-float(prior["uiis_test_foreground_mIoU_percent"]))<1e-7
        b0.append(dict(arm=arm, training_condition=B0_NAMES[arm],
                       synthetic_images=int(prior["synthetic_images"]), real_images=int(prior["real_images"]),
                       seed_17_miou_percent=results[0], seed_29_miou_percent=results[1], seed_43_miou_percent=results[2],
                       three_seed_mean_percent=np.mean(results), seed_sd_pp=np.std(results, ddof=1)))
    return b0


def matched_rows(scores, b0, meta):
    bzero = {r["arm"]:r for r in b0}
    records = []
    for r in scores:
        if r["cohort"]!="full":
            continue
        stage=r["stage"]
        val=r["foreground_miou_percent"]
        b = bzero.get(stage,{})
        records.append(dict(stage=stage, training_stage=STAGE_NAMES[stage], seed=meta[stage]["seed"],
                            b4_miou_percent=val, paper_table3_uiis_miou_percent=PAPER[stage],
                            b4_minus_paper_pp=val-PAPER[stage], b0_reference=stage if b else "",
                            b0_seed17_miou_percent=b.get("seed_17_miou_percent",""),
                            b0_three_seed_mean_percent=b.get("three_seed_mean_percent",""),
                            b4_minus_b0_mean_pp=val-b["three_seed_mean_percent"] if b else "",
                            b4_minus_b0_seed17_pp=val-b["seed_17_miou_percent"] if b else ""))
    return records


def chart(figure_dir, comparison):
    fig, ax = plt.subplots(figsize=(8.7,3.4))
    y = np.array([2,1,0])
    for i, row in enumerate(comparison):
        val, paper = row["b4_miou_percent"], row["paper_table3_uiis_miou_percent"]
        ax.plot([val,paper],[y[i],y[i]],color="#cbd5da",lw=2,zorder=1)
        ax.scatter([val],[y[i]],color="#08695e",s=63,zorder=3)
        ax.scatter([paper],[y[i]],s=55,marker="s",facecolor="white",edgecolor="#40345f",lw=1.8,zorder=3)
        if row["b0_reference"]:
            ax.scatter([row["b0_three_seed_mean_percent"]],[y[i]],marker="^",s=60,color="#b46a19",zorder=3)
    ax.set_yticks(y, [STAGE_NAMES[r["stage"]] for r in comparison])
    ax.set_xlim(0,100); ax.set_ylim(-.62,2.65); ax.set_xlabel("Six-class UIIS mean IoU (%)")
    ax.grid(axis="x", color="#e2e7e9", lw=.8); ax.tick_params(axis="y",length=0)
    ax.spines[["left","top","right"]].set_visible(False)
    markers=[Line2D([],[],marker="o",ls="",color="#08695e",label="B4 measured"),
             Line2D([],[],marker="s",ls="",color="#40345f",mfc="white",label="B4 Table 3"),
             Line2D([],[],marker="^",ls="",color="#b46a19",label="B0 three-seed mean")]
    ax.legend(handles=markers,ncol=3,loc="lower left",bbox_to_anchor=(0,1),frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir/"uiis-comparison.png",dpi=175,bbox_inches="tight"); plt.close(fig)


def examples(study, manifest, ids, sel):
    rows = read(manifest)
    assert [r["id"] for r in rows] == list(ids)
    # Fix examples solely using reference labels and source-overlap flags.
    # This illustrates three reference strata without selecting on model error.
    choices = []
    for name, reference_class, keep in (("Fish",2,sel["no_generator_match"]),
                 ("Divers",5,sel["no_uiis_match"]),("Robots",6,sel["full"])):
        qualified=[i for i,row in enumerate(rows) if keep[i] and str(reference_class) in row["classes"].split("|")]
        select = min(qualified,key=lambda i: hashlib.sha256(("b4-qualitative-v1/"+ids[i]).encode()).digest())
        choices.append((name,select))
    palette=np.array([[0,0,0],[213,90,67],[85,218,136],[88,147,241],[239,185,67],[186,104,207],[92,210,221]],dtype=float)
    fig, axes=plt.subplots(len(choices),5,figsize=(13,7),squeeze=False)
    for idx,title in enumerate(("Photograph", "Reference", "Real only", "SynTIDE", "SynTIDE → UIIS")):
        axes[0,idx].set_title(title,fontsize=10)
    selection=[]
    for k,(stratum,i) in enumerate(choices):
        row=rows[i]
        with Image.open(row["image"]) as src:
            rgb=src.convert("RGB")
            rgb.thumbnail((360,240), Image.Resampling.LANCZOS)
            thumb=np.asarray(rgb)
        with Image.open(row["mask"]) as src:
            reference=np.array(src.resize(rgb.size,Image.Resampling.NEAREST))
        ref_ignore=reference==255
        masks=[reference]
        for stage in STAGES:
            with Image.open(study/f"predictions/{stage}/{i:05d}.png") as src:
                masks.append(np.array(src.resize(rgb.size,Image.Resampling.NEAREST)))
        for j,ax in enumerate(axes[k]):
            canvas=thumb.copy()
            if j:
                mask=masks[j-1]
                valid=(mask>0)&(mask<7)&~ref_ignore
                canvas[valid]=.35*canvas[valid]+.65*palette[mask[valid]]
                canvas[ref_ignore]=.3*canvas[ref_ignore]+.7*np.array([165,168,169])
            ax.imshow(canvas); ax.set(xticks=[],yticks=[])
            for sp in ax.spines.values(): sp.set_visible(False)
        axes[k,0].set_ylabel(stratum, fontsize=9)
        selection.append(dict(image_id=row["id"], group=row["group"], reference_stratum=stratum,
                              no_detected_uiis_train_match=int(sel["no_uiis_match"][i]),
                              no_detected_generator_match=int(sel["no_generator_match"][i])))
    labels=CLASSES[1:]+("Ignore",)
    handles=[Line2D([],[],color=(palette[i+1]/255 if i<6 else "#a5a8a9"),lw=5,label=label)
             for i,label in enumerate(labels)]
    fig.legend(handles=handles,loc="lower center",ncol=7,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1),pad=.5)
    fig.savefig(study/"figures/uiis-predictions.jpg",dpi=150,pil_kwargs={"quality":87},bbox_inches="tight")
    plt.close(fig)
    save(study/"metrics/displayed-examples.csv",selection)


def write_table(caption, columns, records):
    def escaped(x):
        return html.escape(str(x))
    header="".join(f"<th scope='col'>{escaped(x)}</th>" for x in columns)
    body=[]
    for r in records:
        first,*nums=r
        body.append("<tr><th scope='row'>"+escaped(first)+"</th>"+"".join("<td>"+escaped(v)+"</td>" for v in nums)+"</tr>")
    return "<div class='scroll'><table><caption>"+escaped(caption)+"</caption><thead><tr>"+header+"</tr></thead><tbody>"+"".join(body)+"</tbody></table></div>"


def embed(path, alt):
    kind = "png" if path.suffix == ".png" else "jpeg"
    return "<img alt='"+html.escape(alt,quote=True)+"' src='data:image/"+kind+";base64,"+base64.b64encode(path.read_bytes()).decode("ascii")+"'>"


def fmt(v, signed=False):
    if v=="" or v is None:
        return "—"
    digits=4 if signed and 0<abs(v)<.005 else 2
    return format(v,("+" if signed else "")+"."+str(digits)+"f").replace("-", "−")


def anchored_link(path, label):
    return "<a href='"+html.escape(str(path),quote=True)+"'>"+html.escape(label)+"</a>"


def relative(path, parent):
    return os.path.relpath(path,start=parent)


def mask_sensitivity(study, comparison):
    records = read(study/"metrics/author-style-iou.csv")
    alternate = {r["stage"]:r for r in records if r["family"].lower()=="b4"}
    audit = json.loads((study/"reproduction/author-style-conversion.json").read_text())
    assert audit["training_images_checked"] == audit["exact_raw_seven_foreground_and_seafloor_plus_background_masks"] == 3937
    assert audit["disagreeing_pixels"] == 0
    assert set(alternate) == set(STAGES)
    cells = []
    for row in comparison:
        stage = row["stage"]
        other = alternate[stage]
        assert other["images"] == "691" and other["seed"] == str(row["seed"])
        primary = row["b4_miou_percent"]
        assert abs(primary-float(other["primary_miou_percent"])) < 1e-5
        alternative = float(other["foreground_miou_percent"])
        cells.append([STAGE_NAMES[stage],fmt(primary),fmt(alternative),fmt(alternative-primary,True),
                      fmt(PAPER[stage]),fmt(alternative-PAPER[stage],True)])
    primary_delta=float(alternate["finetuned"]["primary_miou_percent"])-float(alternate["synthetic"]["primary_miou_percent"])
    alternate_delta=float(alternate["finetuned"]["foreground_miou_percent"])-float(alternate["synthetic"]["foreground_miou_percent"])
    table = write_table("Same predictions with two UIIS reference masks (%)",
        ["Training path","B0-study references","Release-style references","Change (pp)","Table 3","Release-style − Table 3 (pp)"],cells)
    changed = audit["validation_disagreement_on_primary_valid_pixels"]
    restored = audit["pixels_ignored_by_primary_but_scored_by_secondary"]
    pixels = audit["compared_native_training_pixels"]
    paragraph = (f"The polygon conversion used for this sensitivity check reproduced every pixel in the 3,937 released UIIS training masks "
                 f"({pixels:,} pixels). Applying it to the 691 public held-out instance annotations changed {changed:,} previously scored reference "
                 f"pixels and assigned a class to {restored:,} pixels previously marked ignore. All predictions and photographs are identical across the two columns. "
                 f"Fine-tuning changed mIoU by {fmt(primary_delta,True)} points under the primary policy and {fmt(alternate_delta,True)} under the reconstructed policy. "
                 "Agreement on the training release supports this reconstruction. The authors’ downstream test PNGs and evaluator remain unavailable.")
    return table + "<p>" + paragraph + " " + anchored_link(MASKS,"Author discussion of the conversion") + ".</p>"


def precision_history(study):
    modes={"amp_fp16":"mixed FP16","fp32_decoder":"FP16 encoder and FP32 decoder","fp32":"full FP32"}
    recovery=json.loads((study/"runs/synthetic/recovery.json").read_text())
    assert recovery["all_model_and_optimizer_tensors_verified_finite"]
    explanations=[]
    final=[]
    scaled_retries=0
    for s in STAGES:
        run=study/"runs"/s
        record=json.loads((run/"complete.json").read_text())
        final.append(STAGE_NAMES[s]+", "+modes[record.get("precision_mode_at_completion","amp_fp16")])
        path=run/"numerical-events.jsonl"
        if not path.is_file():
            continue
        events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for e in events:
            if "fallback_mode" in e:
                explanations.append(f"{STAGE_NAMES[s]} switched to {modes[e['fallback_mode']]} at update {e['step']:,}.")
            if e["issue"]=="scaled_gradient_overflow":
                scaled_retries+=1
    if scaled_retries:
        explanations.append(f"The gradient scaler also retried {scaled_retries:,} overflowed backward "
                            + ("pass." if scaled_retries==1 else "passes."))
    explanation=" ".join(explanations)
    modes_at_finish=" · ".join(final)
    sources=" · ".join([anchored_link("runs/synthetic/recovery.json","Recovery record"),
                        anchored_link("runs/synthetic/numerical-events.jsonl","Synthetic precision events")])
    return (f"During synthetic pretraining, mixed-precision arithmetic overflowed. Training resumed from the verified "
            f"checkpoint at update {recovery['restored_step']:,}, restoring model, optimizer, gradient scaling and random state. "
            f"{explanation} The recovery reused the same augmented batch and preserved the planned sample order, coverage and learning-rate schedule. "
            f"Final training precision by stage was {modes_at_finish}. Evaluation used FP32 for all three stages. {sources}.")


def html_report(study, original, expanded, comparison, scores, classes, effects, b0, meta):
    by_stage = {r["stage"]:r for r in comparison}
    adj = {(r["stage"],r["comparator"],r["cohort"]):r for r in effects}
    score = {(r["stage"],r["cohort"]):r for r in scores}
    dif = adj["finetuned","real","full"]
    gain = dif["difference_pp"]
    verdict = ("The interval includes zero." if dif["lower95_pp"]<=0<=dif["upper95_pp"] else
               "The interval lies above zero." if dif["lower95_pp"]>0 else "The interval lies below zero.")
    intro = (f"After full SynTIDE pretraining and real UIIS fine-tuning, B4 reached <strong>{fmt(by_stage['finetuned']['b4_miou_percent'])}%</strong> "
             f"on the 691-image UIIS evaluation split. The difference from the real-only B4 baseline was <strong>{fmt(gain,True)} percentage points</strong> "
             f"(paired 95% test-group interval {fmt(dif['lower95_pp'],True)} to {fmt(dif['upper95_pp'],True)}). {verdict} Table 3 reports a +5.20-point difference.")
    vs_pretrain=adj["finetuned","synthetic","full"]
    ft_delta=fmt(vs_pretrain["difference_pp"],True)
    study_vs_b0 = (f"On the full UIIS evaluation split, our B4 real-only score differed from the B0 three-seed mean by "
                   f"{fmt(by_stage['real']['b4_minus_b0_mean_pp'],True)} points. For synthetic pretraining the difference was "
                   f"{fmt(by_stage['synthetic']['b4_minus_b0_mean_pp'],True)} points.")
    master=write_table("UIIS foreground mIoU (%)",["Training path","Our B4","B4 Table 3","Difference (pp)","B0 mean · 3 seeds","B0 · seed 17"],
           [[STAGE_NAMES[r["stage"]],fmt(r["b4_miou_percent"]),fmt(r["paper_table3_uiis_miou_percent"]),fmt(r["b4_minus_paper_pp"],True),
             fmt(r["b0_three_seed_mean_percent"]),fmt(r["b0_seed17_miou_percent"])] for r in comparison])
    old=write_table("Archived B0 training conditions", ["B0 training condition","Synthetic images","Real images","Mean mIoU (%)","Seed SD (pp)","Seed 17 (%)"],
              [[r["training_condition"],f"{r['synthetic_images']:,}",f"{r['real_images']:,}",fmt(r["three_seed_mean_percent"]),fmt(r["seed_sd_pp"]),fmt(r["seed_17_miou_percent"])] for r in b0])
    group_rows=[]
    for c,title in GROUPS.items():
        d=adj["finetuned","real",c]
        group_rows.append([title,str(score["real",c]["images"])]+[fmt(score[s,c]["foreground_miou_percent"]) for s in STAGES]+
                          [fmt(d["difference_pp"],True),fmt(d["lower95_pp"],True)+" to "+fmt(d["upper95_pp"],True)])
    screened=write_table("Sensitivity to visual matches with training sources",["Evaluation cohort","Images","Real only","SynTIDE","SynTIDE → UIIS","Gain (pp)","95% interval (pp)"],group_rows)
    cl={(r["stage"],r["cohort"],r["class_id"]):r for r in classes}
    class_gains = {i:cl["finetuned","full",i]["iou_percent"]-cl["real","full",i]["iou_percent"] for i in range(1,7)}
    most,least=max(class_gains,key=class_gains.get),min(class_gains,key=class_gains.get)
    extremes=(f"Relative to real-only B4, the largest class-level change after synthetic pretraining and real fine-tuning "
              f"was {fmt(class_gains[most],True)} points for {CLASSES[most].lower()}. The smallest was {fmt(class_gains[least],True)} "
              f"points for {CLASSES[least].lower()}.")
    fine_effects={i:cl["finetuned","full",i]["iou_percent"]-cl["synthetic","full",i]["iou_percent"] for i in range(1,7)}
    most_fine,least_fine=max(fine_effects,key=fine_effects.get),min(fine_effects,key=fine_effects.get)
    stage_detail=(f"Fine-tuning the synthetic checkpoint changed {CLASSES[most_fine].lower()} IoU by {fmt(fine_effects[most_fine],True)} "
                  f"points and {CLASSES[least_fine].lower()} IoU by {fmt(fine_effects[least_fine],True)}. These changes partially cancel in the six-class mean.")
    per_class=write_table("Category overlap on the official 691 images (%)", ["Category","Images containing class","Real only","SynTIDE","SynTIDE → UIIS"],
        [[c,cl["real","full",i]["images_with_class"]]+[fmt(cl[s,"full",i]["iou_percent"]) for s in STAGES]
          for i,c in enumerate(CLASSES) if i])
    stage_recipe=[]
    for s in STAGES:
        path = study / "runs" / s / "progress.csv"
        last = read(path)[-1]
        unique = 54_510 if s=="synthetic" else 3_937
        realized = int(last["sample_presentations"])
        stage_recipe.append([STAGE_NAMES[s],f"{unique:,}",f"{realized:,}",f"{realized/unique:.2f}",
                             f"{meta[s]['updates']:,}", meta[s]["input_side"], meta[s]["seed"]])
    recipe=write_table("Downstream runs in this study",["Training stage","Images","Presentations","Mean passes","Updates","Input long side","Seed"],stage_recipe)
    audit = json.loads((study / "reproduction/filter-audit.json").read_text())
    noise_ignored = float(audit["filtered_pixel_fraction"]) * 100
    assert audit["teacher_sha256"] == meta["real"]["checkpoint_sha256"]
    source_pixels=np.array(audit["source_pixels_by_class"],dtype=np.int64)
    kept_pixels=np.array(audit["retained_pixels_by_class"],dtype=np.int64)
    assert len(source_pixels)==len(kept_pixels)==len(CLASSES) and (source_pixels>0).all()
    assert (source_pixels>=kept_pixels).all() and int(source_pixels.sum())==audit["synthetic_pixels"]
    retained=100*kept_pixels/source_pixels
    lowest=1+int(np.argmin(retained[1:]))
    retention=write_table("Generated target pixels retained after filtering",["Class","Raw pixels","Retained pixels","Retained (%)"],
                          [[name,f"{src:,}",f"{kept:,}",fmt(ratio)] for name,src,kept,ratio in zip(CLASSES,source_pixels,kept_pixels,retained)])
    covered = int(audit["synthetic_images"]-audit["fully_ignored_images"])
    batch_audit = json.loads((study/"reproduction/filtered-batch-coverage.json").read_text())
    assert batch_audit["all_scheduled_batches_supervised"]
    synthetic_batches=sum(e["batches"] for e in batch_audit["epochs"])
    assert synthetic_batches == meta["synthetic"]["updates"]
    train_audit = json.loads((study/"reproduction/author-training-mask-audit.json").read_text())
    train_agreement = float(train_audit["agreement_percent_on_valid_pixels"])
    scorefig=embed(study/"figures/uiis-comparison.png","Observed B4 and B0 mean IoU compared with the B4 values in TIDE Table 3")
    ex=embed(study/"figures/uiis-predictions.jpg","Three UIIS scenes with their ground truth and outputs from each B4 stage")
    orig_report=anchored_link(relative(expanded/"report.html",study),"Four-condition B0 report")
    files=" · ".join(anchored_link(path,label) for path,label in (
        ("metrics/stage-comparison.csv","B4 and Table 3"),("metrics/b0-reference.csv","B0 scores and seeds"),
        ("metrics/class-iou.csv","Class scores and support"),("metrics/cohort-scores.csv","Holdout scores"),
        ("metrics/stage-differences.csv","Paired stage comparisons"),("metrics/displayed-examples.csv","Illustrated image IDs"),
        ("metrics/author-style-iou.csv","Reference-mask sensitivity, B4 and B0"),("reproduction/author-style-conversion.json","Reference-mask conversion audit"),
        ("reproduction/filter-audit.json","Pixel-filter audit"),("reproduction/README.txt","Reproduction instructions"),
        ("training-targets/UIIS-labels.zip","UIIS label archive"),("training-targets/SynTIDE-B4-filtered.zip","Filtered SynTIDE target archive"),
        ("figures/uiis-comparison.png","Score figure"),("figures/uiis-predictions.jpg","Prediction examples")))
    model_links = " · ".join(anchored_link("runs/"+s+"/model/model.safetensors",STAGE_NAMES[s]) for s in STAGES)
    paper_links = " · ".join([anchored_link(TABLE3,"TIDE Table 3"),anchored_link(AUTHOR,"Author clarification of pretraining")])
    filter_links = " · ".join([anchored_link(FREEMASK,"FreeMask paper"),anchored_link(FILTER_CODE,"FreeMask reference implementation")])
    source_links = " · ".join([anchored_link(SETTINGS,"Author clarification of synthetic scale"),anchored_link(train_audit["reference"],"Released generator-training data")])
    clean=adj["finetuned","real","no_generator_match"]
    masks = mask_sensitivity(study,comparison)
    precision = precision_history(study)
    style="""body{margin:0;color:#21383a;background:white;font:16px/1.55 system-ui,sans-serif}main{box-sizing:border-box;max-width:1080px;margin:auto;padding:38px clamp(14px,4vw,48px) 68px}header{border-top:5px solid #08695e;padding-top:22px}h1{font-size:clamp(32px,5vw,54px);line-height:1.08;max-width:22ch;letter-spacing:-.025em}h2{margin:1.5em 0 .45em;font-size:clamp(23px,3vw,30px);line-height:1.14}p{max-width:88ch}.lead{font-size:20px}.scroll{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;font-size:14px;width:100%;font-variant-numeric:tabular-nums}caption{text-align:left;font-weight:bold;margin-bottom:6px}th,td{padding:7px 10px;border-bottom:1px solid #d7e1df;text-align:right}thead th{background:#edf3f2}th:first-child{text-align:left}tbody th{font-weight:500}a{color:#0b665b}img{max-width:100%;height:auto}figure{margin:18px 0}figcaption,.notes{font-size:14px;color:#4b6263}section{border-top:1px solid #d7e1df;margin-top:26px}footer{margin-top:30px;border-top:1px solid #c8d3d2}"""
    body=f"""<header><h1>SegFormer-B4 on underwater scenes</h1><p class='lead'>{intro}</p></header>
<section><h2>Comparison with TIDE Table 3</h2>{master}
<p>Fine-tuning changed the synthetic checkpoint’s UIIS mIoU by {ft_delta} points in this study and −1.10 points in Table 3. All measured B4 values use one seed. B0 means use seeds 17, 29 and 43. Table 3 reports aggregate scores without per-frame predictions or uncertainty. Differences from the paper are descriptive. {paper_links}.</p>
<figure>{scorefig}</figure></section>
<section><h2>How much does mask conversion affect mIoU?</h2>{masks}</section>
<section><h2>Comparison with the earlier B0 runs</h2>{old}
<p>{study_vs_b0} Each B0 run used 4,000 updates with a batch of eight. The two B0 mixes presented real and synthetic images together on every update. The B4 fine-tuning stage starts with the trained synthetic checkpoint and then uses UIIS exclusively. There is no directly staged B0 comparator. Changes in backbone, available examples, source overlap, training budget and label processing prevent assigning the score difference to model size alone. {orig_report}.</p></section>
<section><h2>Evaluation-set overlap</h2>{screened}
<p>The official UIIS training set includes photographs that match 349 of the 691 evaluation images under the archived visual screen. These training images are retained here to follow the official split, and the B4 baseline and fine-tuning stage train on the full split. The B0 real-data runs used duplicate-screened subsets. An additional screen against all published TIDE generator training sources flags 578 evaluation images. On the 113 remaining images, SynTIDE → UIIS differed from real-only B4 by {fmt(clean['difference_pp'],True)} points, compared with {fmt(gain,True)} on the full split. These 113 images contain only one robot image and three wreck images. The interval for this subset uses {clean['draws_with_all_six_classes']:,} of 2,000 resamples that retain all six reference classes. Both screens can miss strongly transformed photographs.</p></section>
<section><h2>Which classes account for the score?</h2>{per_class}
<p>{extremes} {stage_detail} IoU pools correctly and incorrectly labeled pixels across all test images for each category. Each of the six foreground categories then contributes equal weight. Valid background affects foreground false positives, and ignored reference pixels are excluded.</p>
<figure>{ex}<figcaption>Examples use a fixed hash within three predefined reference-label strata. The first row passes the full generator-source screen and the second passes the original UIIS-train screen. Colored overlays show foreground predictions. Gray marks reference pixels excluded from scoring.</figcaption></figure></section>
<section><h2>Training and validation</h2>{recipe}
<p>The real-only baseline and SynTIDE model each start from an ImageNet-pretrained MiT-B4 encoder and a new seven-class segmentation head. The trained real-only baseline scores pixels for the FreeMask-style filter. For each class, synthetic target pixels are ignored when teacher cross-entropy exceeds 1.5 times its mean on real training pixels of that class, following the released FreeMask code. The FreeMask paper describes calibration on synthetic pixels. TIDE specifies a tolerance of 1.5 and leaves the calibration source open. The filter excluded {noise_ignored:.2f}% of synthetic pixels. The synthetic-pretraining checkpoint uses generated targets with an indirect dependency on human labels through its teacher. Fine-tuning initializes from the complete synthetic checkpoint, including its segmentation head. {filter_links}.</p>
{retention}<p>Of 54,510 generated images, {covered:,} retained at least one pixel that could contribute to the training loss. Every one of the {synthetic_batches:,} training batches contained supervised pixels. Among the foreground categories, {CLASSES[lowest].lower()} had the lowest retained fraction at {retained[lowest]:.2f}%.</p>
<p>The real training split contains all 3,937 official UIIS images and uses the semantic masks distributed in the TIDE generator-training archive. Their sea-floor ID 7 maps to background 0. After that conversion the masks agree with {train_agreement:.2f}% of pixels considered valid under the earlier labeling policy. The synthetic pool contains all 54,510 released pairs, ten variations per caption. Primary evaluation uses the same 691 original UIIS validation image identifiers and derived reference masks as the earlier B0 study. Single-scale input is aspect-fitted onto a square padded canvas. The evaluator removes padding, upsamples logits to each original image size, and takes the class with the highest logit per pixel. All 2,073 exported prediction PNGs were independently rescored.</p>
<p>Each synthetic pass used 6,813 batches of eight and a final batch of six. Real-data batches contained eight images. All stages used AdamW with weight decay 0.01, a 100-update warmup and linear decay from peak learning rates of 0.00006 for the encoder and 0.0006 for the decoder. Horizontal flips and mild brightness changes augmented the inputs. The real-only and fine-tuned runs matched their UIIS update budget. Each stage used its final checkpoint, independently of test mIoU.</p>
<p>{precision}</p>
<p>Stage differences pair predictions on the same visual groups in 2,000 bootstrap draws. They describe variability across sampled test groups for the trained checkpoints. Training-seed uncertainty requires additional B4 runs. The released Table 3 materials establish the backbone, split, label categories, synthetic pretraining direction and dataset scale. Their downstream update budget and exact author-converted semantic test masks remain unavailable.</p></section>
<footer><h2>Results and reproduction files</h2><p>{files}</p><p>Trained checkpoints &nbsp; {model_links}</p><p>{source_links}.</p></footer>"""
    dest=study/"report.html"
    tmp=dest.with_suffix(".tmp")
    tmp.write_text("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>B4 UIIS segmentation comparison</title><style>"+style+"</style></head><body><main>"+body+"</main></body></html>")
    tmp.replace(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,default=Path("outputs/segformer-study-b4"))
    p.add_argument("--original",type=Path,default=Path("outputs/segformer-study"))
    p.add_argument("--expanded",type=Path,default=Path("outputs/segformer-study-expanded"))
    p.add_argument("--manifest",type=Path,default=Path("work/segstudy-b4/manifests/uiis_test.csv"))
    a=p.parse_args()
    mats, ids, groups, meta=load_arrays(a.study)
    sel=selections(a.original,ids)
    scores,classes,effects=summarize(mats,groups,sel)
    b0=load_b0(a.original,a.expanded,ids,groups)
    comparison=matched_rows(scores,b0,meta)
    directory=a.study/"metrics"
    for name, records in (("stage-comparison",comparison),("b0-reference",b0),("cohort-scores",scores),
                          ("class-iou",classes),("stage-differences",effects)):
        save(directory/(name+".csv"),records)
    (a.study/"figures").mkdir(parents=True,exist_ok=True)
    chart(a.study/"figures",comparison)
    examples(a.study,a.manifest,ids,sel)
    sensitivity = a.study/"metrics/author-style-iou.csv"
    if not sensitivity.exists():
        subprocess.run([sys.executable,"-Bu",str(Path(__file__).with_name("score_author_style.py")),
                        "--study",str(a.study),"--original",str(a.original),"--expanded",str(a.expanded),
                        "--manifest",str(a.study/"reproduction/uiis_author_style_validation.csv"),"--require-complete"],check=True)
    html_report(a.study,a.original,a.expanded,comparison,scores,classes,effects,b0,meta)
    print("Saved",a.study/"report.html")


if __name__=="__main__":
    main()
