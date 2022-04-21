import glob
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np

import utils as u

plt.switch_backend("agg")
import argparse
import warnings

from scipy import ndimage


def resample_image(
    image,
    spacing,
    new_spacing,
    order=3,
    border_mode="nearest",
    outside_val=0,
    prefilter=True,
):
    """Resamples the image to a new voxel spacing using spline interpolation"""
    resampling_factors = tuple(o / n for o, n in zip(spacing, new_spacing))

    if prefilter:
        factor = 0.5 if type(prefilter) is bool else float(prefilter)
        sigmas = tuple((n / o) * factor for o, n in zip(spacing, new_spacing))
        image = ndimage.gaussian_filter(
            image, sigmas, mode=border_mode, cval=outside_val
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ndimage.zoom(
            image, resampling_factors, order=order, mode=border_mode, cval=outside_val
        )


def resample_mask(mask, spacing, new_spacing):
    """Resamples a label mask to a new voxel spacing using nearest neighbor interpolation"""
    return resample_image(
        mask, spacing, new_spacing, order=0, border_mode="constant", prefilter=False
    )


def projections(im, coordinate):
    """
    Make projections of three slices from the image to check orientation
    :param im: image
    :param coordinate: slice number in each image axis that will be shown
    :return: projection of three slices in 2D
    """
    axial = im[coordinate[0]]
    coronal = im[:, coordinate[1], :]
    sagittal = im[:, :, coordinate[2]]

    full_im_shape = (
        axial.shape[0] + coronal.shape[0],
        axial.shape[1] + sagittal.shape[0],
    )
    full_im = np.zeros(full_im_shape)
    full_im[: axial.shape[0], : axial.shape[1]] = axial
    full_im[axial.shape[0] :, : axial.shape[1]] = coronal
    full_im[: axial.shape[0], axial.shape[1] :] = sagittal.T
    ret = full_im
    return ret


def checkDataOrientationandResample(data_dir, store_proj, new_voxel_size, gt_available):
    """
    Function to check the orientation of the data (images) and resample them to voxel-size [1.5, 1.5, 2.5] #[x,y,z]
    :param data_dir: Directory where the data is stored
    :param save_dir: Directory where projections will be stored
    :param store_proj: When True, projections will be made and stored
    :return:
    """

    labfolders = os.listdir(data_dir)  # folders in which the data is stored
    for folder in labfolders:
        if "case" in folder or "Patient_" in folder:  # found a new case/patient
            print("Analyzing image: ", folder)
            if "SegThor" in data_dir:
                data_dirfol = data_dir + "/" + folder + "/"
                im_file = glob.glob(path.join(data_dirfol + folder + ".nii.gz"))[0]
                if gt_available:
                    segm_file = glob.glob(
                        path.join(data_dirfol + "GT" + folder[-3:] + ".nii.gz")
                    )[0]
            elif "KiTS19" in data_dir:
                data_dirfol = data_dir + "/" + folder + "/"
                im_file = glob.glob(path.join(data_dirfol + "imaging.nii.gz"))[0]
                if gt_available:
                    segm_file = glob.glob(
                        path.join(data_dirfol + "segmentation.nii.gz")
                    )[0]
            im, spacing, offset = u.loadImage(im_file)
            if gt_available:
                segm, sp, _ = u.loadImage(segm_file)

            # make sure segmentation and image have the same shape and spacing
            if gt_available:
                assert im.shape == segm.shape
                assert spacing == sp

            # make projections of the data
            if store_proj:
                ret = projections(
                    im, (im.shape[0] // 2, im.shape[1] // 2, im.shape[2] // 2)
                )
                plt.imshow(ret, cmap="gray")
                plt.savefig(data_dir + str(folder) + "image.png")
                plt.close()
                if gt_available:
                    ret = projections(
                        segm,
                        (segm.shape[0] // 2, segm.shape[1] // 2, segm.shape[2] // 2),
                    )
                    plt.imshow(ret)
                    plt.savefig(data_dir + str(folder) + "segmentation.png")
                    plt.close()

            if spacing != new_voxel_size:
                # resample and save the new image
                im = resample_image(im, spacing, new_voxel_size)
                if gt_available:
                    segm = resample_mask(segm, spacing, new_voxel_size)

            if "SegThor" in data_dir:
                u.saveImage(
                    data_dirfol + folder + "_rs.nii.gz",
                    im,
                    spacing=new_voxel_size[::-1],
                )
                if gt_available:
                    u.saveImage(
                        data_dirfol + "GT" + folder[-3:] + "_rs.nii.gz",
                        segm,
                        spacing=new_voxel_size[::-1],
                    )
            elif "KiTS19" in data_dir:
                u.saveImage(
                    data_dir + "/" + folder + r"/imaging_rs.nii.gz",
                    im,
                    spacing=new_voxel_size,
                )
                if gt_available:
                    u.saveImage(
                        data_dir + "/" + folder + r"/segmentation_rs.nii.gz",
                        segm,
                        spacing=new_voxel_size,
                    )


if __name__ == "__main__":
    description = (
        "Check orientation of the images and resample them to given voxelsize. "
        "Resampled images will be stored where the original data is stored. Filenames end with '_rs'"
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        type=str,
        help='Path + directory where data is stored. Example: r"/home/julia/data"',
    )
    parser.add_argument(
        "-p",
        "--store_proj",
        required=False,
        type=bool,
        default=False,
        help="Boolean, if True projections of the data will be made and stored in data_dir.",
    )
    parser.add_argument(
        "-gt",
        "--groundtruth_available",
        required=False,
        type=int,
        default=1,
        help="Boolean, if True ground truth will also be resampled.",
    )
    parser.add_argument(
        "-vs",
        "--voxel_size",
        required=True,
        type=float,
        nargs="+",
        default=False,
        help="Tuple containing new voxel_sizes. "
        "For SegThor: 0.9765620231628418, 0.9765620231628418, 2.5; For KiTS19: 1.5 1.5 2.5",
    )
    args = parser.parse_args()

    checkDataOrientationandResample(
        data_dir=args.data_dir,
        store_proj=args.store_proj,
        new_voxel_size=args.voxel_size,
        gt_available=args.groundtruth_available,
    )
