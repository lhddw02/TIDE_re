"""Cross-check the delivered report, archived comparisons, labels and lineage."""
import argparse
import base64
import csv
import hashlib
import io
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def record(path):
    return json.loads(path.read_text())


def match(actual, expected):
    np.testing.assert_allclose(float(actual), float(expected), rtol=0, atol=2e-8)


def mean_iou(cms):
    pooled = cms.sum(axis=0)
    true_positive = pooled.diagonal()
    union = pooled.sum(axis=1) + pooled.sum(axis=0) - true_positive
    assert np.all(union[1:] > 0)
    return float(np.mean(true_positive[1:] / union[1:]) * 100)


class ReportLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.figures = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "a":
            self.hrefs.append(attr["href"])
        if tag == "img":
            self.figures.append(attr["src"])


def verify(study):
    repro = study / "reproduction"
    metrics = study / "metrics"
    baseline = record(repro / "baseline-report-checksums.json")
    for name, sha in baseline.items():
        assert digest(study.parent.parent / name) == sha, name
    stages = ("real", "synthetic", "finetuned")
    paper = dict(zip(stages, (70.2, 76.5, 75.4)))
    training = record(repro / "training-verification.json")
    assert training["total_successful_updates"] == 28442
    assert training["real_and_synthetic_exact_same_initial_model"]
    assert training["finetuned_starts_from_the_exact_complete_synthetic_model"]
    comparisons = {r["stage"]: r for r in read(metrics / "stage-comparison.csv")}
    assert set(comparisons) == set(stages)
    official_ids = [r["id"] for r in read(repro / "manifests/uiis_test.csv")]
    flags = {r["id"]: r for r in read(study.parent / "segformer-study/reproduction/exposure/generator-overlap-flags.csv") if r["split"] == "uiis_test"}
    selectors = {"full": np.ones(len(official_ids), dtype=bool),
                 "no_uiis_match": np.array([int(flags[i]["matched_uiis"]) == 0 for i in official_ids]),
                 "no_generator_match": np.array([int(flags[i]["known_overlap"]) == 0 for i in official_ids])}
    assert [sel.sum() for sel in selectors.values()] == [691, 342, 113]
    scores = {(r["cohort"], r["stage"]): r for r in read(metrics / "cohort-scores.csv")}
    stage_scores = {}
    for s in stages:
        row = comparisons[s]
        completion = record(study / "runs" / s / "complete.json")
        pred = record(metrics / s / "verified.json")
        assert pred["verified_masks"] == len(list((study / "predictions" / s).glob("*.png"))) == 691
        assert digest(study / "runs" / s / "model/model.safetensors") == pred["checkpoint_sha256"] == completion["checkpoint_sha256"]
        with np.load(metrics / s / "confusions.npz", allow_pickle=False) as conf:
            assert conf["id"].tolist() == official_ids
            cm = conf["cm"]
            assert cm.shape == (691, 7, 7)
            for cohort, select in selectors.items():
                independent = mean_iou(cm[select])
                match(scores[cohort, s]["foreground_miou_percent"], independent)
                stage_scores[cohort, s] = independent
        match(row["b4_miou_percent"], stage_scores["full", s])
        match(row["paper_table3_uiis_miou_percent"], paper[s])
        match(row["b4_minus_paper_pp"], stage_scores["full", s] - paper[s])
        match(row["b4_miou_percent"], pred["foreground_miou_percent"])
    for row in read(metrics / "stage-differences.csv"):
        k = row["cohort"]
        match(row["difference_pp"], stage_scores[k, row["stage"]] - stage_scores[k, row["comparator"]])
        assert float(row["lower95_pp"]) <= float(row["upper95_pp"])

    historical = {r["arm"]: r for r in read(study.parent / "segformer-study-expanded/metrics/headline.csv")}
    for r in read(metrics / "b0-reference.csv"):
        match(r["three_seed_mean_percent"], historical[r["arm"]]["uiis_test_foreground_mIoU_percent"])
    alternate = read(metrics / "author-style-iou.csv")
    assert len(alternate) == 15
    for r in alternate:
        path = metrics / "author-style" / ("{family}-{stage}-{seed}.npz".format(**r))
        with np.load(path, allow_pickle=False) as alt:
            assert alt["id"].tolist() == official_ids
            match(r["foreground_miou_percent"], mean_iou(alt["cm"]))
        match(r["delta_pp"], float(r["foreground_miou_percent"]) - float(r["primary_miou_percent"]))
        if r["family"] == "B4":
            match(r["primary_miou_percent"], stage_scores["full", r["stage"]])
    model_filter = record(repro / "filter-audit.json")
    assert model_filter["teacher_sha256"] == training["stages"]["real"]["weight_file_sha256"]
    assert model_filter["filtered_manifest_sha256"] == digest(repro / "manifests/synthetic-filtered.csv")
    assert model_filter["synthetic_images"] == 54510 and model_filter["synthetic_pixels"] == 54510*512*512
    uiis_archive = record(repro / "uiis-mask-bundle.json")
    syn_archive = record(repro / "filtered-target-archive.json")
    assert syn_archive["all_retained_values_equal_raw_source_pixels"] and syn_archive["every_archived_mask_checksum_verified"]
    for name, count, expected in (("UIIS-labels.zip", 5319, uiis_archive["archive_sha256"]),
                                  ("SynTIDE-B4-filtered.zip", 54510, syn_archive["sha256"])):
        path = study / "training-targets" / name
        assert digest(path) == expected
        with ZipFile(path) as archive:
            assert sum(n.endswith(".png") for n in archive.namelist()) == count
    assert record(repro / "author-style-conversion.json")["disagreeing_pixels"] == 0

    report = study / "report.html"
    html = report.read_text()
    links = ReportLinks()
    links.feed(html)
    local = [urlsplit(href) for href in links.hrefs if not urlsplit(href).scheme]
    for target in local:
        assert (study / unquote(target.path)).exists(), target.path
    assert len(links.figures) >= 2
    for src in links.figures:
        header, content = src.split(",", 1)
        assert header.startswith("data:image/") and header.endswith("base64")
        with Image.open(io.BytesIO(base64.b64decode(content, validate=True))) as im:
            im.verify()
    for s in stages:
        assert format(stage_scores["full", s], ".2f") in html
    return {"verified": True, "b4_training_stages": 3, "b4_prediction_masks": 2073,
            "previous_B0_files_with_unchanged_sha256": len(baseline),
            "independently_recomputed_primary_stage_cohort_scores": len(stage_scores),
            "independently_recomputed_secondary_scores": len(alternate),
            "local_report_links_checked": len(local), "embedded_report_figures_checked": len(links.figures),
            "verified_training_archives": 2, "report_sha256": digest(report)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=Path("outputs/segformer-study-b4"))
    args = p.parse_args()
    outcome = verify(args.study)
    dest = args.study / "reproduction/package-verification.json"
    dest.write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))
