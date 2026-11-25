SegFormer-B4 on UIIS and SynTIDE

Open ../report.html for the measured comparisons. The results directory is
independent of the two archived B0 studies. All paths below are relative to
the B4 study folder unless a command specifies a project-relative path.

Where to find the results

  metrics/stage-comparison.csv       B4 and published Table 3, one row per stage
  metrics/b0-reference.csv           All four B0 conditions, including each seed
  metrics/stage-differences.csv     Paired intervals on the same groups of photos
  metrics/cohort-scores.csv         Full split and both overlap-screened subsets
  metrics/class-iou.csv             IoU and test-set support for each category
  metrics/author-style-iou.csv      Same 15 sets of predictions, second reference
  runs/{real,synthetic,finetuned}/model/
                                   Loadable model weights and config at each stage
  predictions/{real,synthetic,finetuned}/
                                   Native-resolution, indexed PNG predictions
  training-targets/UIIS-labels.zip   All 3,937 real training labels, both sets of
                                   691 evaluation labels and per-file hashes
  training-targets/SynTIDE-B4-filtered.zip
                                   All 54,510 filtered labels and per-file hashes

Table 3 uses SegFormer-B4 and reports mean IoU across six foreground classes.
We compare the published UIIS column to the same 691 official validation images
used in our B0 experiments. We pool all valid pixels by class and then average
the six foreground IoUs equally. Class 0 includes background and sea floor.
A foreground prediction on a valid background pixel counts as a false positive.
Class 255 in the reference excludes that pixel. There are no test-time flips or
multiscale predictions. The prior B0 joint mixtures have a different training
sequence from the B4 pretraining and fine-tuning experiment.

Two evaluation references

The primary masks are frozen from the B0 study. They treat crowd pixels and
conflicting foreground targets as ignore. The second reference is an independent
rendering of UIIS's published polygons. Its rendering rules matched all
1,353,146,800 pixels in the authors' released generator-training masks before
merging sea floor with background. We used these rules on the validation polygons
and scored the same saved predictions again. The authors' downstream evaluation
masks were unavailable. Both references are in UIIS-labels.zip. The conversion
audit and its Pillow version are recorded in author-style-conversion.json.

Reproduce this experiment

Use a fresh results folder when repeating training. The output scripts check
existing stage signatures and the filter binds to a specific real checkpoint.
The completed checkpoints can be evaluated or loaded directly without repeating
training. The recorded environment uses PyTorch 2.0.1+cu118, Transformers 4.41.2,
Pillow 10.2.0 and an NVIDIA RTX 3090 Ti. Run the examples from the project root.

  R=outputs/segformer-study-b4/reproduction
  W=work/segstudy-b4
  python "$R/get_encoder.py" --output "$W/pretrained/mit-b4"

DATA-AND-FILTER.txt documents the two upstream image releases, the raw and
filtered manifests, and the filter and archive commands. TRAINING.txt gives the
training commands with explicit input and output directories. For a clean rerun,
prepare the manifest lists, train stage real, run filter_synthetic.py on that
checkpoint, train stage synthetic, then train stage finetuned. Record the final
model from each stage. Each training stage's recipe.json documents its sampling,
augmentation, optimizer and initialization. The local choice was three complete
passes over SynTIDE and 4,000 updates for each real-data stage. The synthetic
student's weights start from the same ImageNet-initialized model as the baseline.
Its targets are filtered by the real-only B4, so real labels influence that stage
through the teacher. The final UIIS stage starts from the entire synthetic model.
See protocol.json for the paper, author clarifications and implementation sources.

The saved manifests contain absolute source paths for this machine. On a new
machine, preserve the original image IDs and order, point the RGB paths to the
source releases and point the label paths to the matching archive sections.
The real archive members use the original UIIS filename stems. The synthetic
members use the exact ID in synthetic-filtered.csv. Point the primary UIIS
validation mask paths to validation-primary, and the secondary ones to
validation-inferred. A changed path changes the manifest checksum. Create new
records for a relocated run and keep the published records as provenance.
Reproducing the B0 comparison also requires the two neighboring B0 output
folders, including their predictions and per-image confusion matrices.

Verify the completed models, re-score and regenerate the report using the
original project layout and environment:

  python "$R/verify_training.py"
  python "$R/score_stages.py" --device cpu --threads 5
  python "$R/check_package.py"

verify_training.py checks the encoder, head, update counts, synthetic epoch
coverage, checkpoint digests and fine-tuning lineage. score_stages.py checks
each primary prediction against its UIIS reference and generates the report.
evaluate_b4.py accepts --redo to regenerate the predictions from a checkpoint.
score_author_style.py scores the second reference. check_package.py checks the
comparison table, report links, archive digests and the original B0 result files.
