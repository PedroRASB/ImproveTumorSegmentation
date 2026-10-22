#!/usr/bin/env python3
"""Lesion mask refinement by nnInteractive voting (see README.md).

For every case and lesion type:
  1. starting mask: the report-refined mask when one exists (R-Super r_super_pseudo_masks.py output); otherwise the
     largest connected component of the model's binary prediction; empty if the report says the organ is healthy.
  2. nnInteractive re-segments every component of the starting mask three times: from 3 interior clicks, from
     scribbles on 3 planes and from boxes on 3 planes (each component is its own nnInteractive object).
  3. vote: keep the voxels on which at least 3 of the 4 masks (starting mask + 3 nnInteractive masks) agree.

Example (one process per GPU):
  python refine_masks.py --pred_root <preds> --refined_root <refined> --meta <report csv> --source_ct <cts> \
      --output_folder <out> --gpu 0 --parts 8 --part 0
"""
import argparse
import csv
import math
import os
import sys
import time
import traceback

import nibabel as nib
import numpy as np
import pandas as pd

MIN_CC_VOX, MAX_CC = 10, 15                 # components that get prompted (largest first); smaller specks are dropped
VOTE_PROMPTS = ('pt3', 'scrib_3p', 'bbox_3p')
VOTE_MIN = 3                                 # keep voxels in >= 3 of {starting mask, clicks, scribbles, boxes}
MODEL_ID = 'nnInteractive_v1.0'


# ------------------------------------------------------------------------------------------------- image helpers
def _canonical(path, dtype=np.float32):
    """Load a NIfTI in RAS orientation (as nnInteractive was trained); returns (original image, array, spacing)."""
    img = nib.load(path)
    can = nib.as_closest_canonical(img)
    return img, np.asanyarray(can.dataobj).astype(dtype), np.array(can.header.get_zooms()[:3], float)


def _to_original(arr_can, orig_img):
    t = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt('RAS'), nib.io_orientation(orig_img.affine))
    return nib.orientations.apply_orientation(arr_can, t)


# ------------------------------------------------------------------------------------------------- prompts
def _interior_points(crop, spacing, n=3):
    """Deepest voxel (distance transform) + farthest-point samples among voxels at least half as deep."""
    from scipy import ndimage as ndi
    edt = ndi.distance_transform_edt(np.pad(crop, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    pts = [np.array(np.unravel_index(np.argmax(edt), edt.shape))]
    cand = np.argwhere(edt >= 0.5 * edt.max())
    d = np.full(len(cand), np.inf)
    while len(pts) < n and len(cand) > 1:
        d = np.minimum(d, np.linalg.norm((cand - pts[-1]) * spacing, axis=1))
        if d.max() <= 2 * max(spacing):
            break
        pts.append(cand[np.argmax(d)])
    return pts


def _plane_prompts(crop, off):
    """On the component's largest axial, coronal and sagittal slice: its 2D box and its skeleton (scribble)."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    out = []
    for ax in (2, 1, 0):
        s = int(np.argmax(crop.sum(axis=tuple(a for a in range(3) if a != ax))))
        sl = np.take(crop, s, axis=ax)
        nz = np.argwhere(sl)
        lo, hi = nz.min(0), nz.max(0) + 1
        b = [None] * 3
        b[ax] = [int(off[ax] + s), int(off[ax] + s + 1)]
        for k, a in enumerate(a for a in range(3) if a != ax):
            b[a] = [int(off[a] + lo[k]), int(off[a] + hi[k])]
        filled = ndi.binary_fill_holes(sl)[lo[0]:hi[0], lo[1]:hi[1]]
        skel = skeletonize(filled)
        if not skel.any():
            skel = filled
        out.append(dict(bbox=b, scribble=np.expand_dims(skel.astype(np.uint8), ax)))
    return out


# ------------------------------------------------------------------------------------------------- nnInteractive
def new_session(model_dir=None):
    """nnInteractive session with the model loaded (downloads nnInteractive_v1.0 on first use if no folder given)."""
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    if model_dir is None:
        from nnInteractive.model_management import ensure_model_available
        model_dir = str(ensure_model_available(MODEL_ID))
    s = nnInteractiveInferenceSession(device=torch.device('cuda:0'), use_torch_compile=False, verbose=False,
                                      torch_n_threads=12, do_autozoom=True, interactions_storage='tensor',
                                      enable_undo=False)
    s.initialize_from_trained_model_folder(model_dir)
    return s


def set_image(s, ct_path):
    import torch
    _, ct, spacing = _canonical(ct_path)
    s.set_image(ct[None])
    s.set_target_buffer(torch.zeros(ct.shape, dtype=torch.uint8))
    return ct.shape, spacing


def _components(mask, largest_only=False):
    """26-connected components with >= MIN_CC_VOX voxels, largest first: [(crop, offset, slices)]."""
    import cc3d
    lab, n = cc3d.connected_components(mask > 0, connectivity=26, return_N=True)
    if n == 0:
        return []
    st = cc3d.statistics(lab)
    order = [k for k in np.argsort(-st['voxel_counts'][1:]) + 1 if st['voxel_counts'][k] >= MIN_CC_VOX]
    order = order[:1] if largest_only else order[:MAX_CC]
    return [(lab[st['bounding_boxes'][k]] == k, np.array([x.start for x in st['bounding_boxes'][k]]),
             st['bounding_boxes'][k]) for k in order]


def _prompt_component(s, crop, off, spacing, prompt):
    """One nnInteractive object grown from one component; returns its segmentation (bool, full image)."""
    s.reset_interactions()
    if prompt == 'pt3':
        for p in _interior_points(crop, spacing, 3):
            s.add_point_interaction(tuple(int(v) for v in off + p), include_interaction=True)
    elif prompt == 'bbox_3p':
        for p in _plane_prompts(crop, off):
            s.add_bbox_interaction(p['bbox'], include_interaction=True)
    elif prompt == 'scrib_3p':
        for p in _plane_prompts(crop, off):
            s.add_scribble_interaction(p['scribble'], include_interaction=True, interaction_bbox=p['bbox'])
    else:
        raise ValueError(prompt)
    return s.target_buffer.numpy() > 0


def vote_refine(s, mask, spacing, largest_only=False):
    """Vote refinement of a binary mask (RAS array; the CT must already be set in session `s`).
    Returns (refined bool mask, {'start', 'pt3', 'scrib_3p', 'bbox_3p'} member masks)."""
    comps = _components(mask, largest_only)
    start = np.zeros(mask.shape, bool)
    for crop, _, sl in comps:
        start[sl] |= crop
    members = {'start': start}
    for prompt in VOTE_PROMPTS:
        grown = np.zeros(mask.shape, bool)
        for crop, off, _ in comps:
            grown |= _prompt_component(s, crop, off, spacing, prompt)
        members[prompt] = grown
    count = np.zeros(mask.shape, np.uint8)
    for m in members.values():
        count += m
    return count >= VOTE_MIN, members


# ------------------------------------------------------------------------------------------------- batch CLI
FIELDS = ['case_id', 'lesion', 'report_count', 'source', 'n_components', 'start_vox', 'clicks_vox', 'scribbles_vox',
          'boxes_vox', 'vote_vox', 'dice_vote_vs_start', 'seconds']


def report_count(meta, cid, lesion):
    """Number of lesions the report lists for the organ (column 'number of <organ> lesion instances'); None if unknown."""
    if meta is None or cid not in meta.index:
        return None
    organ = lesion.replace('_lesion', '')
    col = f"number of {organ.replace('_', ' ').replace('adrenal', 'adrenal gland')} lesion instances"
    if col not in meta.columns:
        return None
    v = meta.loc[cid, col]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return None if pd.isna(v) else float(v)


def work_list(a):
    pairs = None
    if a.ids:
        df = pd.read_csv(a.ids)
        idc = next(c for c in ('BDMAP ID', 'BDMAP_ID', 'case_id', 'id') if c in df.columns)
        if 'lesion' in df.columns:
            pairs = df.groupby(idc)['lesion'].apply(list).to_dict()
            ids = sorted(pairs)
        else:
            ids = sorted(df[idc].unique())
    else:
        ids = sorted(d for d in os.listdir(a.pred_root) if os.path.isdir(os.path.join(a.pred_root, d)))
    chunk = math.ceil(len(ids) / a.parts)
    out = []
    for cid in ids[a.part * chunk:(a.part + 1) * chunk]:
        if pairs is not None:
            lesions = pairs[cid]
        elif a.lesions:
            lesions = a.lesions
        else:
            d = os.path.join(a.pred_root, cid, 'predictions')
            lesions = sorted(f[:-7] for f in os.listdir(d) if f.endswith('_lesion.nii.gz')) if os.path.isdir(d) else []
        out.append((cid, lesions))
    return out


def save(arr, ref_img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), ref_img.affine), path)


def main():
    ap = argparse.ArgumentParser(description='Lesion mask refinement by nnInteractive voting')
    ap.add_argument('--pred_root', required=True, help='predictions: <id>/predictions/<lesion>.nii.gz (binary)')
    ap.add_argument('--refined_root', default=None, help='report-refined masks: <id>/refined_<lesion>_mask.nii.gz')
    ap.add_argument('--source_ct', required=True, help='CTs: <id>/ct.nii.gz')
    ap.add_argument('--output_folder', required=True)
    ap.add_argument('--meta', default=None,
                    help='report CSV ("BDMAP ID" + "number of <organ> lesion instances"); reported-absent -> empty mask')
    ap.add_argument('--lesions', nargs='*', default=None, help='lesion classes (default: all *_lesion files)')
    ap.add_argument('--ids', default=None, help='CSV of case ids; optional "lesion" column limits the lesions')
    ap.add_argument('--model_dir', default=None, help='nnInteractive model folder (default: download/cache v1.0)')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--save_members', action='store_true', help='also save the 4 masks that vote')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)  # before torch is imported (lazily)

    meta = None
    if a.meta:
        meta = pd.read_csv(a.meta, low_memory=False).rename(columns={'BDMAP_ID': 'BDMAP ID'})
        meta = meta.drop_duplicates('BDMAP ID').set_index('BDMAP ID')
    work = work_list(a)
    os.makedirs(a.output_folder, exist_ok=True)
    summary = os.path.join(a.output_folder, f'summary_part{a.part}.csv')
    new_file = not os.path.exists(summary)
    fh = open(summary, 'a', newline='')
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    print(f'part {a.part}/{a.parts}: {len(work)} cases', flush=True)

    session = None
    for n, (cid, lesions) in enumerate(work):
        t0 = time.time()
        todo = [l for l in lesions if a.overwrite or not os.path.exists(os.path.join(a.output_folder, cid, f'{l}.nii.gz'))]
        if not todo:
            continue
        jobs = []  # (lesion, report count, source, starting-mask path)
        for l in todo:
            cnt = report_count(meta, cid, l)
            pred = os.path.join(a.pred_root, cid, 'predictions', f'{l}.nii.gz')
            ref = os.path.join(a.refined_root, cid, f'refined_{l}_mask.nii.gz') if a.refined_root else None
            if cnt == 0:
                jobs.append((l, cnt, 'report says absent', None))
            elif ref and os.path.exists(ref) and np.any(np.asanyarray(nib.load(ref).dataobj)):
                jobs.append((l, cnt, 'refined', ref))
            elif os.path.exists(pred) and np.any(np.asanyarray(nib.load(pred).dataobj)):
                jobs.append((l, cnt, 'prediction', pred))
            else:
                jobs.append((l, cnt, 'nothing to refine', None))
        try:
            if any(j[3] for j in jobs):
                if session is None:
                    session = new_session(a.model_dir)
                shape, spacing = set_image(session, os.path.join(a.source_ct, cid, 'ct.nii.gz'))
            for l, cnt, source, path in jobs:
                t1 = time.time()
                out_path = os.path.join(a.output_folder, cid, f'{l}.nii.gz')
                row = dict(case_id=cid, lesion=l, report_count=cnt, source=source)
                if path is None:
                    geom = os.path.join(a.pred_root, cid, 'predictions', f'{l}.nii.gz')
                    ref_img = nib.load(geom if os.path.exists(geom) else os.path.join(a.source_ct, cid, 'ct.nii.gz'))
                    save(np.zeros(ref_img.shape[:3], np.uint8), ref_img, out_path)
                    row.update(n_components=0, vote_vox=0)
                else:
                    img, mask, _ = _canonical(path, np.uint8)
                    if mask.shape != shape:
                        raise ValueError(f'{path}: shape {mask.shape} differs from the CT {shape}')
                    largest_only = source == 'prediction'
                    vote, mem = vote_refine(session, mask, spacing, largest_only=largest_only)
                    save(_to_original(vote, img), img, out_path)
                    if a.save_members:
                        for k, m in mem.items():
                            save(_to_original(m, img), img, os.path.join(a.output_folder, cid, 'members', f'{l}_{k}.nii.gz'))
                    st = mem['start']
                    row.update(n_components=len(_components(mask, largest_only)), start_vox=int(st.sum()),
                               clicks_vox=int(mem['pt3'].sum()), scribbles_vox=int(mem['scrib_3p'].sum()),
                               boxes_vox=int(mem['bbox_3p'].sum()), vote_vox=int(vote.sum()),
                               dice_vote_vs_start=round(2 * np.logical_and(st, vote).sum() / max(st.sum() + vote.sum(), 1), 4))
                row['seconds'] = round(time.time() - t1, 1)
                writer.writerow(row)
                fh.flush()
        except Exception as e:
            traceback.print_exc()
            for l, cnt, source, _ in jobs:
                if not os.path.exists(os.path.join(a.output_folder, cid, f'{l}.nii.gz')):
                    writer.writerow(dict(case_id=cid, lesion=l, report_count=cnt, source=f'error: {e}'))
            fh.flush()
        print(f'[{n + 1}/{len(work)}] {cid}: {len(todo)} lesion(s) in {time.time() - t0:.1f}s', flush=True)
    fh.close()


if __name__ == '__main__':
    main()
