# Lesion mask refinement with nnInteractive and Report-based refinement

Refines tumour masks with [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive), a
promptable 3D segmentation model, and with the report-based mask refinement from R-Super and Merlin Plus.

In summary: the code takes a tumor mask (per-voxel tumor probabilities), and adjusts it to match the tumor size, number, and locations in radiology reports. Then, it applies nnInteractive to improve tumor borders.

## How it works

For every case and lesion type:

1. **Starting mask** – the report-refined mask if there is one (R-Super `r_super_pseudo_masks.py`); otherwise the
   largest connected component of the model's binary prediction. If the report says the organ has no lesion
   (`--meta`), an empty mask is written.
2. **Three nnInteractive masks** – every component of the starting mask is re-segmented three times, prompted with
   3 interior clicks, with scribbles on 3 planes, and with boxes on 3 planes.
3. **Vote** – a voxel is kept if at least 3 of the 4 masks (starting mask + 3 nnInteractive masks) contain it.

Clicks let nnInteractive regrow under-segmented lesions, boxes and scribbles preserve good shapes, and the vote removes
the leaks of any single prompt. On manually annotated test cases it improved both R-Super models we tried (mean Dice):

| | no-mask model | model trained with masks |
|---|---|---|
| prediction | 0.152 | 0.338 |
| vote refinement | **0.257** | **0.391** |

## Install

```bash
conda create -n nninteractive python=3.11 -y && conda activate nninteractive
pip install torch --index-url https://download.pytorch.org/whl/cu126   # PyTorch with CUDA
pip install -r requirements.txt
```

The nnInteractive weights (`nnInteractive_v1.0`) are downloaded automatically on the first run (into
`~/.nninteractive`, or pass `--model_dir`). Note: the weights are licensed CC BY-NC-SA 4.0 (non-commercial).

## Run

Input layout (BDMAP style; `<lesion>` like `bladder_lesion`):

```
<pred_root>/<id>/predictions/<lesion>.nii.gz      binary prediction
<refined_root>/<id>/refined_<lesion>_mask.nii.gz  report-refined mask (optional)
<source_ct>/<id>/ct.nii.gz                        CT
```

```bash
python refine_masks.py --pred_root <pred_root> --refined_root <refined_root> --source_ct <source_ct> \
    --meta <report_metadata.csv> --output_folder <out>
```

Several GPUs: `run_example.sh` (one process per GPU via `--gpu/--parts/--part`). Other options: `--lesions`,
`--ids` (CSV of cases, optional `lesion` column), `--save_members` (also save the 4 voting masks), `--overwrite`.
Finished lesions are skipped, so interrupted runs can be resumed.

## Output

* `<out>/<id>/<lesion>.nii.gz` – refined mask (same geometry as the input mask).
* `<out>/summary_part<k>.csv` – one row per lesion: which starting mask was used, report count, voxel counts of the
  four masks and of the vote, `dice_vote_vs_start`.

## Notes

* About 10 s per lesion on an RTX 6000 Ada; ~8 GB GPU and ~10 GB RAM per process.
* It refines where it is prompted: a lesion the starting mask misses completely cannot be recovered.
* nnInteractive results can differ by a few voxels between runs (cuDNN autotuning); more for very large lesions.
