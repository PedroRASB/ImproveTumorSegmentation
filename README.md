# From AI tumor probabilities to refined masks

This code refines AI-made tumor masks in two steps: first with the radiology report, then with a promptable 3D segmentation model.

**AI tumor probabilities → R-Super report refinement → nnInteractive → final tumor masks**

1. **Match the report.** `report_refinement.py` uses the report-based refinement from [R-Super](https://github.com/MrGiovanni/R-Super) and [Merlin Plus](https://github.com/MrGiovanni/MerlinPlus) to adjust the predicted tumors toward the reported size, number, and organ location.
2. **Improve the borders.** `refine_masks.py` automatically prompts [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) with points, scribbles, and boxes. The final mask keeps voxels supported by at least three of four masks: the report-refined mask and the three nnInteractive results.

Both steps run **in sequence**. Start with per-voxel probabilities from your tumor segmentation AI; the code produces the intermediate and final masks.

## Install

Requires a CUDA GPU.

```bash
conda create -n mask_refine python=3.11 -y
conda activate mask_refine
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

nnInteractive weights download automatically on first use. Its model weights use the [CC BY-NC-SA 4.0 license](https://github.com/MIC-DKFZ/nnInteractive#license).

## Prepare the inputs

Example for bladder tumors (`<id>` such as `BDMAP_00000001`):

```text
predictions/<id>/predictions_raw/bladder_lesion.nii.gz  # AI tumor probabilities, 0–1
predictions/<id>/predictions/bladder.nii.gz             # binary organ mask
cts/<id>/ct.nii.gz                                     # CT image
```

The preprocessing command below resamples the CT, probabilities, and masks onto a shared **1 × 1 × 1 mm grid** for report-based sizing. Originals are kept unchanged. Use these two CSVs, prepared from the radiology reports:

| File | Required columns |
| --- | --- |
| `meta.csv` — one row per case | `BDMAP ID`, `number of bladder lesion instances` |
| `reports.csv` — one row per tumor | `BDMAP_ID`, `Standardized Organ`, `Standardized Location`, `Tumor Size (mm)` |

Use organ names such as `bladder`, sizes such as `12 x 10 x 8` (mm), and `u` for unknown size or location. Include every input case in `meta.csv`.

## Run both steps

From this repository, prepare the 1 mm inputs, then run the two refinement steps in order:

```bash
python resample_inputs.py \
    --pred_root predictions --source_ct cts --output_folder inputs_1mm && \
python report_refinement.py \
    --pred_root inputs_1mm/predictions --source_ct inputs_1mm/cts --meta meta.csv --reports reports.csv \
    --classes bladder --output_folder report_refined --out kept_ids.txt --gpu 0 && \
python refine_masks.py \
    --pred_root inputs_1mm/predictions --source_ct inputs_1mm/cts --meta meta.csv \
    --refined_root report_refined --lesions bladder_lesion --output_folder final_masks --gpu 0
```

`resample_inputs.py` uses linear interpolation for CTs and probabilities, and nearest-neighbor interpolation for masks. It also accepts probability `.npz` files (`arr_0`) on the original CT grid and writes NIfTI files.

For other organs, change `--classes`, `--lesions`, filenames, and the lesion-count column together. For several GPUs, edit the paths, organ lists, and GPU count in [run_example.sh](run_example.sh), then run `bash run_example.sh`; it runs preprocessing and both refinement steps automatically.

Final masks are saved to `final_masks/<id>/bladder_lesion.nii.gz` on the **1 mm grid**. Intermediate masks and report scores are in `report_refined/`; voting summaries are in `final_masks/summary_part*.csv`. A summary source of `refined` confirms that nnInteractive used the report-refined mask.

Existing outputs are skipped; add `--overwrite` to all three commands to recompute.

<details>
<summary>Missing report sizes and other limitations</summary>

If a report size is unknown or no nonempty report-refined mask exists, nnInteractive falls back to the largest component of `predictions/<id>/predictions/bladder_lesion.nii.gz`, if supplied. Without either starting mask, it writes an empty mask. A report count of zero also produces an empty mask.

Report matching is approximate: an oversized prediction can remain unchanged. Inspect the masks and summaries; a `source=error: ...` row signals a failed case even if the command exits normally.

</details>

## Papers

The report-based refinement comes from [R-Super](https://github.com/MrGiovanni/R-Super) and [Merlin Plus](https://github.com/MrGiovanni/MerlinPlus); the 3D border refinement uses [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive).

**Merlin Plus: A Large-Scale, Multi-cancer, Image-Mask-Report Dataset** <br/>
Pedro R. A. S. Bassi†, Wenxuan Li†, Szymon Płotka†, Ruby Honjol, Jakub Prządo, Xinze Zhou, Kang Wang, Yang Yang, Malte Jensen, Akshay S. Chaudhari, Curtis P. Langlotz, Alan L. Yuille, and Zongwei Zhou. <br/>
*MICCAI 2026, LNCS 16895. Springer Nature Switzerland.* <br/>
<a href="https://papers.miccai.org/miccai-2026/paper/4063_paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-purple" alt="Merlin Plus paper PDF"></a>

**Learning Segmentation from Radiology Reports** <br/>
Pedro R. A. S. Bassi, Wenxuan Li, Jieneng Chen, Zheren Zhu, Tianyu Lin, Sergio Decherchi, Andrea Cavalli, Kang Wang, Yang Yang, Alan Yuille, Zongwei Zhou. <br/>
*Johns Hopkins University* <br/>
MICCAI 2025, Best Paper Award (runner-up) <br/>
<a href="https://www.cs.jhu.edu/~zongwei/publication/bassi2025learning.pdf"><img src="https://img.shields.io/badge/Paper-PDF-purple" alt="R-Super paper PDF"></a>

**Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks** <br/>
Pedro R. A. S. Bassi, Xinze Zhou, Wenxuan Li, Szymon Płotka, Jieneng Chen, Qi Chen, Zheren Zhu, Jakub Prządo, Ibrahim E. Hamacı, Sezgin Er, Yuhan Wang, Ashwin Kumar, Bjoern Menze, Jarosław B. Ćwikła, Yuyin Zhou, Akshay S. Chaudhari, Curtis P. Langlotz, Sergio Decherchi, Andrea Cavalli, Kang Wang, Yang Yang, Alan L. Yuille, Zongwei Zhou. <br/>
*Johns Hopkins University, Stanford University, University of Zurich, University of Warsaw, Italian Institute of Technology, UCSF* <br/>
Preprint, 2025 <br/>
<a href="https://www.cs.jhu.edu/~zongwei/publication/bassi2025scaling.pdf"><img src="https://img.shields.io/badge/Paper-PDF-purple" alt="R-Super scaling paper PDF"></a>

**nnInteractive: Redefining 3D Promptable Segmentation** <br/>
Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. <br/>
Preprint, 2025 (\*: equal contribution) <br/>
<a href="https://arxiv.org/abs/2503.08373"><img src="https://img.shields.io/badge/Paper-arXiv-purple" alt="nnInteractive paper"></a>

## Citations

If you use this code, please cite the corresponding work below. These citations are copied from the project GitHub READMEs.

From [R-Super](https://github.com/MrGiovanni/R-Super#citation):

```bibtex
@inproceedings{bassi2025learning,
  title={Learning Segmentation from Radiology Reports},
  author={Bassi, Pedro R. A. S. and Li, Wenxuan and Chen, Jieneng and Zhu, Zheren and Lin, Tianyu and Decherchi, Sergio and Cavalli, Andrea and Wang, Kang and Yang, Yang and Yuille, Alan L. and Zhou, Zongwei},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={305--315},
  year={2025},
  publisher={Springer},
  doi={10.1007/978-3-032-04971-1_29}
}
@article{bassi2025scaling,
  title={Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks},
  author={Bassi, Pedro R. A. S. and Zhou, Xinze and Li, Wenxuan and P{\l}otka, Szymon and Chen, Jieneng and Chen, Qi and Zhu, Zheren and Prz{\k{a}}do, Jakub and Hamamci, Ibrahim E. and Er, Sezgin and others},
  journal={arXiv preprint arXiv:2510.14803},
  year={2025}
}
```

From [Merlin Plus](https://github.com/MrGiovanni/MerlinPlus#citations):

```bibtex
@InProceedings{BasPed_Merlin_MICCAI2026,
    author = { Bassi, Pedro R. A. S. AND Li, Wenxuan AND Płotka, Szymon AND Honjol, Ruby AND Prządo, Jakub AND Zhou, Xinze AND Wang, Kang AND Yang, Yang AND Jensen, Malte AND Chaudhari, Akshay S. AND Langlotz, Curtis P. AND Yuille, Alan L. AND Zhou, Zongwei},
    title = { { Merlin Plus: A Large-Scale, Multi-cancer, Image-Mask-Report Dataset } },
    booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
    year = {2026},
    publisher = {Springer Nature Switzerland},
    volume = {LNCS 16895},
    month = {September},
    page = {pending}
}
```

From [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive#citation), which provides a formatted citation:

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373
>
> *: equal contribution
