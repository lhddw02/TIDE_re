# Original SUIM photographs used in the exposure screen

The authors published 1,525 training/validation and 110 test photographs. The TIDE generator index names 1,598 distinct SUIM photographs. Filename matching recovered 1,488 from `train_val/images` and all 110 from `TEST/images`. This source was used to check evaluation images for potential generator exposure. [SUIM dataset and split definitions](https://irvlab.cs.umn.edu/resources/suim-dataset) · [Author-hosted archive](https://drive.google.com/file/d/1uEnlqKrlt6lITc_i80NTtb7iHGcO47sU/view)

Download the SUIM archive from the authors. Extract `Training_TIDE_Data/UWDense.json` from the official TIDE training archive. The TIDE filename index uses one JSON object per line. The bundled `generator-rgb-index.jsonl` contains the same filenames and can also be used with the staging script. The SHA-256 in the archived audit identifies the complete original JSON. [TIDE training archive](https://huggingface.co/datasets/hongk1998/TIDE/tree/main)

From the reproduction directory, run the staging script with the local paths to these two files. It preserves the original JPEG bytes, checks each selected ZIP member’s CRC on extraction, and records its SHA-256 digest and split.

```bash
python exposure/stage_suim.py --archive /path/to/SUIM.zip --generator-index /path/to/UWDense.json --out /path/to/suim_refs
```

Pass `/path/to/suim_refs` to the combined exposure screen’s `--suim` option. The archived `suim-source-members.csv` records each selected original filename and image digest. The companion `suim-source-audit.json` records coverage and the hash of the TIDE filename index.

To retrieve only the image portions of the author’s 5,245,090,436-byte ZIP, the following command fetches its directory and RGB members over HTTP byte ranges. Supply the resulting sparse ZIP to the same staging command. On a filesystem supporting sparse files, it uses about 171 MiB of storage.

```bash
python exposure/fetch_suim_photos.py --out /path/to/SUIM-original-rgb.sparse.zip
```
