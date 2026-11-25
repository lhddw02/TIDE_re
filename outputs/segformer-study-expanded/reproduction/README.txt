Expanded SynTIDE and UIIS SegFormer-B0 comparison

Open ../report.html for the four-condition comparison. The original study is
in the sibling segformer-study folder. ../baseline-snapshot.zip contains a
verified backup of that complete folder, including its report, checkpoints,
predictions, metrics and reproduction materials. snapshot-sha256.json inside
the ZIP lists every archived file and its digest. The baseline archive contains
model outputs and manifests. Public source photographs and masks are obtained
as described in the original reproduction/DATA.md.

Experimental design
The combined.csv manifest is the union of the original 3,000-image synthetic-only
and 3,000-image real-only pools. It includes every original 50/50 training image
and adds 1,500 distinct images of each source. The original three training
populations, checkpoints, test images, masks, group IDs and metrics are preserved.

Each of the new runs uses a pretrained ImageNet MiT-B0 encoder, a seven-class
SegFormer head, 512-pixel input, batch size eight, 4,000 optimizer updates and
the paired seeds 17, 29 and 43. Each batch includes four images of each source.
Each seed sees 16,000 synthetic and 16,000 real image presentations, so the
average number of presentations per distinct image falls from 10.67 in the
original 50/50 condition to 5.33 in the expanded condition. Absolute optimizer
budgets and schedules remain comparable. The real-only condition uses the same
3,000 distinct human-labeled images as the expanded condition, with 32,000 real
image presentations per run.

Reproducing the training and evaluation on this machine
Run from the project root. Use the TIDE Python environment documented in the
original reproduction/environment-and-sources.json and the pretrained encoder
identified by reproduction/pretrained-encoder.json in that original folder.
ORIGINAL points to the complete original study. Use a fresh NEW directory for a
new run. The original manifests identify the prepared images and masks. ENCODER
is the locally downloaded nvidia/mit-b0 revision.

  export ORIGINAL=outputs/segformer-study
  export CODE=outputs/segformer-study-expanded/reproduction
  export NEW=work/expanded-rerun
  export ENCODER=work/segstudy/pretrained/mit-b0
  mkdir -p "$NEW/reproduction/manifests"
  python "$CODE/build_combined.py" --original-manifests "$ORIGINAL/reproduction/manifests" --output "$NEW/reproduction/manifests"
  python "$CODE/train.py" --pretrained "$ENCODER" --manifests "$NEW/reproduction/manifests" --output "$NEW/runs"
  for seed in 17 29 43; do
    python "$CODE/evaluate.py" --arm combined --seed "$seed" --device cuda --batch-size 4 --workers 1 --manifests "$ORIGINAL/reproduction/manifests" --output "$NEW"
  done
  python "$CODE/verify_expanded.py" --study "$NEW" --original "$ORIGINAL" --combined "$NEW/reproduction/manifests/combined.csv"
  python "$CODE/check_images.py" --original "$ORIGINAL" --combined "$NEW/reproduction/manifests/combined.csv" --output "$NEW/reproduction/image-audit.json"
  python "$CODE/preserve_baseline.py" --original "$ORIGINAL" --archive "$NEW/baseline-snapshot.zip" --summary "$NEW/reproduction/baseline-snapshot.json"
  python "$CODE/report.py" --previous "$ORIGINAL" --output "$NEW"
  python "$CODE/check_report.py" --original "$ORIGINAL" --expanded "$NEW"

For a different machine, prepare the raw data, derived masks, manifests and
original runs following the original study guide. The saved manifests carry
the machine's original absolute paths. Use the corresponding locally valid
manifests and match their stable image IDs and groups. File hashes of a
regenerated manifest can change when its absolute paths change.

Execution record
Seed 29 resumed at its saved 3,000-update checkpoint after an interrupted
process. The model weights, optimizer, gradient scaler and random-generator
states were restored. The 75 updates after that checkpoint were replayed.
The three replayed 25-update mean losses agreed with the interrupted trace
to within 0.0000045. runs/combined/seed-29/restart.json records both traces.

New artifacts
runs/combined/seed-*/model contains the three final Hugging Face checkpoints.
Each run also contains a frozen recipe, the loss trace and a completion record
with the model digest. predictions/combined contains native-resolution indexed
PNG masks. Each corresponding metrics/combined per-image.csv associates its
original image ID with the saved mask. Each confusions.npz holds per-frame
class-by-class counts and fish-free-frame false-positive counts. All counts
exclude reference ID 255. The original classes and reporting conventions are
documented in the previous study and retained in the new report.

Optional original-folder verification (run from the project root):
  python "$CODE/preserve_baseline.py" --original "$ORIGINAL" --archive outputs/segformer-study-expanded/baseline-snapshot.zip --check
