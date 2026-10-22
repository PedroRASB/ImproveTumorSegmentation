#!/bin/bash
# Refine all cases with one process per GPU. Edit the paths below.
parts=8
pred_root=/path/to/predictions           # <id>/predictions/<lesion>.nii.gz
refined_root=/path/to/report_refined     # <id>/refined_<lesion>_mask.nii.gz  (optional: leave empty to skip)
meta=/path/to/report_metadata.csv        # "BDMAP ID" + "number of <organ> lesion instances" (optional)
source_ct=/path/to/cts                   # <id>/ct.nii.gz
output_folder=/path/to/output

extra=()
[ -n "$refined_root" ] && extra+=(--refined_root "$refined_root")
[ -n "$meta" ] && extra+=(--meta "$meta")
for ((part=0; part<parts; part++)); do
    python refine_masks.py --pred_root "$pred_root" --source_ct "$source_ct" --output_folder "$output_folder" \
        "${extra[@]}" --gpu "$part" --parts "$parts" --part "$part" &
done
wait
