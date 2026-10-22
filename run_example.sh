#!/bin/bash
# Run report refinement, then nnInteractive, with one process per GPU. Edit the paths below.
set -euo pipefail

parts=8
pred_root=/path/to/predictions           # <id>/predictions/<lesion>.nii.gz
refined_root=/path/to/report_refined     # <id>/refined_<lesion>_mask.nii.gz
meta=/path/to/report_metadata.csv        # "BDMAP ID" + "number of <organ> lesion instances"
reports=/path/to/per_tumor_metadata.csv
source_ct=/path/to/cts                   # <id>/ct.nii.gz
prepared_root=/path/to/1mm_inputs
output_folder=/path/to/output
classes=bladder,duodenum,esophagus,gallbladder,prostate,spleen,stomach,uterus
lesions=(bladder_lesion duodenum_lesion esophagus_lesion gallbladder_lesion prostate_lesion spleen_lesion stomach_lesion uterus_lesion)

python resample_inputs.py --pred_root "$pred_root" --source_ct "$source_ct" --output_folder "$prepared_root"
pred_root="$prepared_root/predictions"
source_ct="$prepared_root/cts"

mkdir -p "$refined_root"
workers=()
for ((part=0; part<parts; part++)); do
    python report_refinement.py --pred_root "$pred_root" --source_ct "$source_ct" \
        --meta "$meta" --reports "$reports" --classes "$classes" --output_folder "$refined_root" \
        --out "$refined_root/kept_ids_part${part}.txt" --gpu "$part" --parts "$parts" --part "$part" &
    workers+=("$!")
done
status=0
for worker in "${workers[@]}"; do
    wait "$worker" || status=1
done
if ((status)); then
    exit "$status"
fi

workers=()
for ((part=0; part<parts; part++)); do
    python refine_masks.py --pred_root "$pred_root" --source_ct "$source_ct" --output_folder "$output_folder" \
        --refined_root "$refined_root" --meta "$meta" --lesions "${lesions[@]}" \
        --gpu "$part" --parts "$parts" --part "$part" &
    workers+=("$!")
done
status=0
for worker in "${workers[@]}"; do
    wait "$worker" || status=1
done
exit "$status"
