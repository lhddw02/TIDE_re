SynTIDE and SegFormer-B0 experiment

Open ../report.html for the study results. The relative paths below start in this
reproduction directory. DATA.md describes how to retrieve the original releases,
convert masks, reproduce the splits and check them. Source revisions, file digests
and package versions are recorded in the accompanying JSON files.

Training protocol
Each arm contains 3,000 distinct images. The mixed arm takes 1,500 of each
source from the two pure pools and presents four examples of each per batch.
The final checkpoint is taken after 4,000 optimizer updates. The pretrained
ImageNet-1k encoder, seven-class segmentation head, batch size eight, image size
512, augmentation and optimizer schedule are shared. Seeds 17, 29 and 43 are
paired across arms. The 400-image UIIS development reserve was kept outside
training. No real validation score was used to select the synthetic checkpoint.
The nine recipe.json files contain the complete training settings.

Reproduce the study
The example uses Bash and a CUDA-enabled Python environment matching
environment-and-sources.json, plus the data-preparation packages in DATA.md and
matplotlib. Use a fresh STUDY directory for a new experiment. PREPARED points to
the directory created by the data-preparation steps.

  export PREPARED=/absolute/path/to/prepared-data
  export STUDY=/absolute/path/to/new-study
  export ENCODER=/absolute/path/to/mit-b0
  mkdir -p "$ENCODER" "$STUDY"
  for file in config.json pytorch_model.bin preprocessor_config.json; do
    curl -fsSL "https://huggingface.co/nvidia/mit-b0/resolve/80983a413c30d36a39c20203974ae7807835e2b4/$file" -o "$ENCODER/$file"
  done
  python train.py --pretrained "$ENCODER" --manifests "$PREPARED/manifests" --output "$STUDY/runs"

Compare the three encoder file checksums with pretrained-encoder.json. The
experiment used a 24 GB NVIDIA RTX 3090 Ti. train.py stages the training images
and masks in GPU memory. A different GPU, CUDA kernel or library version can
change the numerical results.

Evaluate the nine frozen models and verify their exported label masks:

  for arm in synthetic mixed real; do
    for seed in 17 29 43; do
      python evaluate.py --arm "$arm" --seed "$seed" --device cuda --batch-size 4 --workers 1 --manifests "$PREPARED/manifests" --output "$STUDY"
    done
  done
  python verify_results.py --study "$STUDY" --manifests "$PREPARED/manifests" --output "$STUDY/verification.json"

The native PNGs in ../predictions store the seven output class IDs. Each
per-image.csv under ../metrics maps IDs to image predictions. The confusions.npz
files contain the 7 x 7 ground-truth-by-prediction matrix for each image in
manifest order. Values for undefined per-image scores are blank.

Rebuild the sensitivity analysis and report on the same public releases:

  mkdir -p "$STUDY/reproduction/manifests"
  cp -R exposure "$STUDY/reproduction/"
  cp manifests/deepfish-habitats.csv "$STUDY/reproduction/manifests/"
  python overlap_sensitivity.py --study "$STUDY" --flags "$STUDY/reproduction/exposure/generator-overlap-flags.csv"
  python fish_size_audit.py --shards "$PREPARED/downloads/coralscapes" --output "$STUDY/metrics/coralscapes-fish-size.json"
  python report.py --manifests "$PREPARED/manifests" --output "$STUDY"

The combined generator-source screen uses published UWDense.json names and
the original RGB photos. exposure/source-index.json identifies the released
TIDE generator index and USIS10K archive. DATA.md covers downstream data.
To repeat the source screen, point UIIS at the prepared UIIS/train image
directory and USIS/SUIM at their image folders or archives. The SUIM images
can be staged using exposure/stage_suim.py. Use a directory containing only the
generator-source photos for each source to avoid alternate copies with the
same basename.

  python generator_overlap.py --index exposure/generator-rgb-index.jsonl --manifests "$PREPARED/manifests" --uiis "$PREPARED/uiis/train/images" --usis10k /path/to/USIS10K.zip --suim /path/to/suim_refs --output "$STUDY/reproduction/exposure"

Stable frame IDs in the source-screen flags permit reuse after relocating the
data. verify_results.py checks the manifest digests stored by the associated
training run. The frozen manifests shipped here retain their original absolute
paths. For a new machine, use freshly prepared manifests for that machine's new
training and evaluation runs.
