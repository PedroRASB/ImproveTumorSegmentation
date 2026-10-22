#!/usr/bin/env python3
"""Prepare a shared 1 mm CT, probability, and mask grid for report refinement."""

import argparse
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
from nibabel.spaces import vox2out_vox
import numpy as np


def load_prediction(path, ct_image):
    if path.suffix != '.npz':
        return nib.load(path)
    with np.load(path) as archive:
        values = archive['arr_0'].astype(np.float32)
    if values.shape != ct_image.shape:
        raise ValueError(f'{path}: NPZ shape {values.shape} differs from CT {ct_image.shape}')
    return nib.Nifti1Image(values, ct_image.affine)


def resample_image(image, target, output_path, order, fill, probabilities=False, overwrite=False):
    if output_path.exists() and not overwrite:
        existing = nib.load(output_path)
        if existing.shape != target[0] or not np.allclose(existing.affine, target[1]):
            raise ValueError(f'{output_path}: existing grid differs; use --overwrite')
        return
    if len(image.shape) != 3:
        raise ValueError(f'{output_path}: expected a 3D image, got {image.shape}')
    if image.header.get_xyzt_units()[0] not in ('mm', 'unknown'):
        raise ValueError(f'{output_path}: input spatial units must be millimeters')
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f'{output_path}: input contains nonfinite values')
    if probabilities and (values.min() < 0 or values.max() > 1):
        raise ValueError(f'{output_path}: probabilities must be in [0, 1]')
    input_image = nib.Nifti1Image(values, image.affine)
    result = resample_from_to(input_image, target, order=order, cval=fill)
    output_dtype = image.get_data_dtype() if order == 0 else np.float32
    output_image = nib.Nifti1Image(np.asanyarray(result.dataobj).astype(output_dtype), target[1])
    output_image.header.set_xyzt_units('mm')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'.{output_path.name}.tmp.nii.gz')
    nib.save(output_image, temporary_path)
    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred_root', required=True, type=Path)
    parser.add_argument('--source_ct', required=True, type=Path)
    parser.add_argument('--output_folder', required=True, type=Path,
                        help='separate output directory containing predictions/ and cts/')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output_root = args.output_folder.resolve()
    for input_root in (args.pred_root.resolve(), args.source_ct.resolve()):
        if (output_root == input_root or output_root.is_relative_to(input_root)
                or input_root.is_relative_to(output_root)):
            parser.error('--output_folder must be separate from the input directories')
    case_dirs = sorted(path for path in args.pred_root.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        ct_image = nib.load(args.source_ct / case_dir.name / 'ct.nii.gz')
        if len(ct_image.shape) != 3:
            raise ValueError(f'{case_dir.name}: expected a 3D CT, got {ct_image.shape}')
        target = vox2out_vox((ct_image.shape, ct_image.affine), voxel_sizes=(1, 1, 1))
        resample_image(ct_image, target, output_root / 'cts' / case_dir.name / 'ct.nii.gz',
                       order=1, fill=-1024, overwrite=args.overwrite)
        for folder in ('predictions', 'predictions_raw'):
            source_dir = case_dir / folder
            if not source_dir.is_dir():
                continue
            for source_path in sorted(source_dir.iterdir()):
                if not source_path.is_file():
                    continue
                if source_path.name.endswith('.nii.gz'):
                    output_name = source_path.name
                elif source_path.suffix in ('.nii', '.npz'):
                    if source_path.with_suffix('.nii.gz').exists():
                        continue
                    if source_path.suffix == '.npz' and source_path.with_suffix('.nii').exists():
                        continue
                    output_name = source_path.stem + '.nii.gz'
                else:
                    continue
                image = load_prediction(source_path, ct_image)
                probabilities = folder == 'predictions_raw'
                output_path = output_root / 'predictions' / case_dir.name / folder / output_name
                resample_image(image, target, output_path, order=1 if probabilities else 0, fill=0,
                               probabilities=probabilities, overwrite=args.overwrite)
        print(f'{case_dir.name}: {ct_image.shape} -> {target[0]} at 1 mm', flush=True)


if __name__ == '__main__':
    main()
