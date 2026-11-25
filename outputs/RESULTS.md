# Does SynTIDE improve underwater segmentation?

On 691 real UIIS validation images, SegFormer-B4 scored **54.52% mIoU** after training on real data, **56.72%** after synthetic pretraining, and **56.82%** after synthetic pretraining followed by real-data fine-tuning. The last stage gained **2.29 percentage points** over real-only training. Its paired 95% interval (−1.90 to +5.86 points) includes zero, so the size and direction of that gain remain uncertain.

UIIS mIoU gives equal weight to wrecks, fish, reefs, aquatic plants, divers and robots. We pooled valid pixels across images before computing each class’s IoU. All B0 and B4 runs used the same UIIS evaluation images and reference masks.

## B4 compared with TIDE Table 3

| Training stage | Our UIIS mIoU | Paper UIIS mIoU | Difference (percentage points) |
|---|---:|---:|---:|
| UIIS real only | 54.52% | 70.2% | −15.68 |
| Full SynTIDE pretraining | 56.72% | 76.5% | −19.78 |
| SynTIDE → UIIS fine-tuning | 56.82% | 75.4% | −18.58 |

We used one B4 seed, three passes over all 54,510 SynTIDE image–mask pairs, and 4,000 updates on 3,937 UIIS training images for each real-data stage. The real-only B4 filtered the synthetic masks, so real labels indirectly influenced pretraining. The paper leaves the downstream training budget unspecified, and we used reconstructed validation masks. These differences limit how precisely we can explain the gap to Table 3. [Training protocol](segformer-study-b4/reproduction/protocol.json) · [Stage scores](segformer-study-b4/metrics/stage-comparison.csv) · [TIDE Table 3](https://arxiv.org/html/2503.21771v2#S5.T3)

Fine-tuning added **0.10 points** over the synthetic checkpoint (paired 95% interval −4.42 to +4.91). That small average hides two large changes. Robot IoU rose from 26.68% to 39.81%, while aquatic-plant IoU fell from 33.71% to 17.25%. On the 113 validation images with no detected visual match to the generator’s training sources, the three stages scored **43.36%, 43.44%, and 44.78%**, respectively. The intervals describe variation across evaluation groups for this training seed. [Paired comparisons](segformer-study-b4/metrics/stage-differences.csv) · [Per-class scores](segformer-study-b4/metrics/class-iou.csv) · [Overlap-screened scores](segformer-study-b4/metrics/cohort-scores.csv)

## B0 training mixtures

Each B0 condition used 4,000 updates per seed. Both mixtures trained on real and synthetic images together. UIIS values show the mean ± standard deviation across three seeds. The two external datasets measure fish IoU only.

| Training condition | Synthetic images | Real images | UIIS mIoU (%) | DeepFish fish IoU (%) | Coralscapes fish IoU (%) |
|---|---:|---:|---:|---:|---:|
| Synthetic only | 3,000 | 0 | 34.93 ± 1.02 | 45.79 | 3.54 |
| Half synthetic, half real | 1,500 | 1,500 | 46.78 ± 0.99 | 53.41 | 4.28 |
| Real only | 0 | 3,000 | **49.30 ± 0.47** | 53.06 | **6.12** |
| Doubled mixture | 3,000 | 3,000 | 48.54 ± 0.84 | **56.49** | 4.91 |

Adding 3,000 synthetic images to the 3,000 real images changed B0 UIIS mIoU by **−0.76 points** at the same update budget. The doubled mixture had the highest DeepFish fish IoU. B0 and B4 also differed in training-set size, mask filtering, and training schedule, so their scores cannot isolate the effect of backbone size. [B0 results across datasets](segformer-study-expanded/metrics/headline.csv) · [B0 paired comparisons](segformer-study-expanded/metrics/differences.csv)

The HTML reports include example predictions and additional metrics. Open them locally after cloning. [Original B0 study](segformer-study/report.html) · [Expanded B0 study](segformer-study-expanded/report.html) · [B4 study](segformer-study-b4/report.html)
