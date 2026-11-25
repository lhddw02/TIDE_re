# Reproducing the segmentation data

The seven frozen CSVs in `manifests/` identify every image and mask used in the comparison. The `id` column is stable across machines. The `image` and `mask` columns contain absolute paths on the experiment machine. The `group` column records a caption, inferred recording, near-duplicate component, or collection site, according to the source. The `classes` column lists the labels present in the mask.

| Manifest | Images | Groups | Purpose |
|---|---:|---:|---|
| synthetic | 3,000 | 3,000 | SynTIDE training |
| mixed | 3,000 | 3,000 | 1,500 SynTIDE and 1,500 UIIS training images drawn from the two pure pools |
| real | 3,000 | 2,225 | UIIS training |
| uiis_dev | 400 | 284 | Development reserve drawn from the official training split |
| uiis_test | 691 | 649 | Full official UIIS validation split used as the common final test |
| deepfish_test | 186 | 49 | Fish transfer test on the official DeepFish segmentation test split |
| coralscapes_test | 473 | 5 | Fish transfer test on the official CoralscapesV2 test sites |

## Retrieve and prepare the data

1. Download the authors’ SynTIDE images, segmentation masks and caption list, and the DeepFish segmentation images, binary masks and split lists. Place them under a common directory as shown below. The TIDE release names its annotation archive `SynTIDE.zip`. The captions are a separate file. [SynTIDE release](https://huggingface.co/datasets/hongk1998/TIDE/tree/main) · [DeepFish release](https://alzayats.github.io/DeepFish/)
2. Set `TIDE_DATA` to the directory containing those two releases. Set `SEGSTUDY_DATA` to a writable directory for derived masks, checks, and newly generated split files. On the experiment machine these directories were `/data/TIDE` and `work/segstudy/data`.
3. From the directory containing this document, run the commands below. They require Python 3.9, curl and an internet connection for the first download. Use a dedicated Python environment.

```text
TIDE_DATA/
  SynTIDE_Caption.json
  SynTIDE/images/
  SynTIDE/semseg_annotations/
  DeepFish/Segmentation/test.csv
  DeepFish/Segmentation/images/
  DeepFish/Segmentation/masks/
```

```bash
export TIDE_DATA=/absolute/path/to/source-data
export SEGSTUDY_DATA=/absolute/path/to/prepared-data
python -m pip install numpy==1.26.4 Pillow==10.2.0 opencv-python-headless==4.10.0.84 pyarrow==17.0.0 pycocotools==2.0.10
python download_sources.py
python data_prep.py
python audit_data.py
```

`download_sources.py` retrieves the original UIIS annotations, matching photograph shards, and the three official CoralscapesV2 test shards. It compares each length and SHA-256 digest with `source-lock.json`. Matching values verify the bytes used here. On a different machine, pass the regenerated directory `$SEGSTUDY_DATA/manifests` to the training script using `--manifests`. Use the image IDs, group IDs, and class counts to compare regenerated splits across machines. The exact file hashes in `integrity.json` incorporate the original absolute paths. [UIIS](https://huggingface.co/datasets/LiamLian0727/UIIS) · [CoralscapesV2](https://huggingface.co/datasets/josauder/coralscapesV2)

| Local source file | SHA-256 |
|---|---|
| SynTIDE.zip | `257332448ca06edb71ca260273775bd8f44f271b814161e34cf8a62fc4f7c303` |
| SynTIDE_Caption.json | `6e9c2a4b51b7c0355b85a33b82bb36e99cb04cf7b9febbf71f2cde1761ae0659` |
| DeepFish.tar | `8acabb8a314fa8eb45fa9cc6829c84bdc127b1f8f605f2e152304e98b6dbb46c` |

## Label definitions and checks

| ID | Seven-class training and UIIS testing | Binary transfer testing |
|---:|---|---|
| 0 | Background and sea floor | Valid non-fish pixels |
| 1 | Wrecks and ruins | |
| 2 | Fish and vertebrates | Fish |
| 3 | Reefs and invertebrates | |
| 4 | Aquatic plants | |
| 5 | Human divers | |
| 6 | Robots | |
| 255 | Excluded from loss and metrics | Excluded from metrics |

SynTIDE supplies integer semantic masks. Each synthetic pair represents a distinct caption selected from 5,451 candidates. The release contains ten image variations per caption. UIIS supplies instance annotations. We rasterized polygons or run-length encodings, united same-class instances, and ignored crowd regions and overlaps between different classes. UIIS unmarked pixels and sea-floor labels map to the shared background. Its masks can omit recognizable animals. DeepFish source foreground value 255 maps to fish. For CoralscapesV2, label 25 maps to fish, label 0 is ignored, and other labeled pixels count as non-fish.

We screened candidate images using a byte digest and two 64-bit visual hashes. A visual match required Hamming distance of four bits or less in both the DCT perceptual hash and the difference hash. Every linked component touching a test image was excluded from the training candidates. The development split reserved whole eligible UIIS components. Class-aware selection then took one photograph per available group before including further distinct frames in the real-only arm. File-level audits confirmed 3,000 distinct decoded RGB photographs in every arm and zero identical decoded photographs across the combined training and evaluation sets. Distinct nearby frames can remain within a training group. The hash thresholds can miss crops or large viewpoint changes.

The `data-checks/` folder contains example synthetic image/mask overlays, prompt checks, real image/mask overlays and sampled duplicate pairs. `manifests/pixel_support.csv` records image and pixel counts for each class. `data_audit.json` and `integrity.json` in the same folder provide counts, split definitions and checksums. `duplicate_matches.csv` lists candidate pairs with their exact or perceptual criterion and distances.

TIDE learned from public underwater photographs, including UIIS. The field `matches_original_uiis_train` in `manifests/uiis_generator_overlap.csv` marks the 349 official holdout frames whose visual component also contains an image from the original UIIS train split. The complete generator-exposure screen resolved all 14,573 filenames across UIIS, USIS10K and SUIM. It flagged 578 of the 691 UIIS test frames, leaving 113 frames for the sensitivity comparison. It found zero matches on the two fish transfer tests under the specified thresholds. The flags are in `exposure/generator-overlap-flags.csv`. Downstream SegFormer training used the separately screened manifests above.
