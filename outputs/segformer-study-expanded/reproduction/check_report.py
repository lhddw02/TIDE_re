"""Verify that the combined report reproduces its headlines and references usable files."""
import argparse
import base64
import csv
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

from preserve_baseline import check as check_original_tree


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.hrefs.append(attrs["href"])
        if tag == "img" and attrs.get("src"):
            self.images.append(attrs["src"])


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_report(original, expanded):
    check_original_tree(original, expanded/"baseline-snapshot.zip")
    checks = json.loads((expanded/"reproduction/verification-expanded.json").read_text())
    assert checks["native_masks_independently_rescored"] == 4050
    assert len(list(expanded.glob("predictions/combined/*/*/*.png"))) == 4050
    assert len(list(expanded.glob("runs/combined/*/model/model.safetensors"))) == 3
    page = References()
    page.feed((expanded / "report.html").read_text())
    local_links = 0
    for ref in page.hrefs + page.images:
        parts = urlsplit(ref)
        if not parts.scheme and parts.path:
            assert (expanded/unquote(parts.path)).exists(), ref
            local_links += 1
    for src in page.images:
        if src.startswith("data:image/"):
            Image.open(BytesIO(base64.b64decode(src.split(",",1)[1]))).verify()
    for image in (expanded/"figures").glob("*"):
        if image.suffix.lower() in (".png",".jpg",".jpeg"):
            with Image.open(image) as im:
                im.verify()
    groups = read_csv(original/"reproduction/exposure/generator-overlap-flags.csv")
    kept = {g["id"] for g in groups if g["split"]=="uiis_test" and g["known_overlap"]=="0"}
    assert len(kept)==113
    summary = {(r["dataset"],r["arm"],r["metric"]):r for r in read_csv(expanded/"metrics/summary.csv")}
    ds = read_csv(expanded/"metrics/differences.csv")
    deltas = {(r["dataset"],r["arm"],r["reference_arm"],r["metric"]):r for r in ds}
    cohorts = ("uiis_test","uiis_without_generator_match","deepfish_test","coralscapes_test")
    means = {}
    paired = {}
    for cohort in cohorts:
        multiclass = cohort.startswith("uiis")
        split = "uiis_test" if multiclass else cohort
        metric = "foreground_mIoU" if multiclass else "fish_IoU"
        means[cohort] = {}
        for arm in ("synthetic","mixed","real","combined"):
            study = expanded if arm=="combined" else original
            scores = []
            for seed in (17,29,43):
                path = study/"metrics"/arm/f"seed-{seed}"/split/"confusions.npz"
                with np.load(path) as data:
                    cms = data["cm"]
                    if cohort=="uiis_without_generator_match":
                        cms = cms[[i in kept for i in data["id"]]]
                    matrix = cms.sum(axis=0)
                tp = matrix.diagonal()
                denominator = matrix.sum(0) + matrix.sum(1) - tp
                ids = np.arange(1,7) if multiclass else [2]
                scores.append(float((tp[ids]/denominator[ids]).mean()))
            report = summary[cohort,arm,metric]
            assert np.isclose(np.mean(scores),float(report["seed_mean"]),rtol=0,atol=1e-9)
            assert np.isclose(np.std(scores,ddof=1),float(report["seed_sd"]),rtol=0,atol=1e-9)
            means[cohort][arm] = 100*np.mean(scores)
        paired[cohort] = {}
        for comparator in ("mixed","real","synthetic"):
            diff = deltas[cohort,"combined",comparator,metric]
            assert abs(float(diff["difference_pp"]) - (means[cohort]["combined"] - means[cohort][comparator])) < 1e-7
            paired[cohort][comparator] = [float(diff[key]) for key in ("difference_pp","lower95_pp","upper95_pp")]
    audit = {"archived_original_unchanged":True, "new_checkpoints":3,"new_predictions_verified":4050,
             "headline_means_recomputed":16,"paired_headline_differences_recomputed":12,
             "local_report_links_verified":local_links,"images_in_html":len(page.images),
             "score_means_percent":means,"new_minus_comparator_pp_and_paired_95pct_interval":paired}
    (expanded/"reproduction/package-verification.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--expanded",type=Path,required=True)
    a=p.parse_args()
    check_report(a.original,a.expanded)
