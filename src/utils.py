import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.utils.data
import tqdm

import data_augmentation as DA


class dice_loss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction="elementwise_mean"):
        super().__init__(size_average, reduce, reduction)

    def forward(self, pred, target):
        nclass = pred.shape[1]
        x = 0
        loss = self.dice_loss_perclass(pred[:, x, :, :], target[:, x, :, :])
        x += 1
        while x < nclass:
            loss_perclass = self.dice_loss_perclass(
                pred[:, x, :, :], target[:, x, :, :]
            )
            loss += loss_perclass
            x += 1
        loss = -1.0 * loss
        return loss

    def dice_loss_perclass(self, pred, target):
        """
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """
        smooth = 1e-07

        # have to use contiguous since they may from a torch.view op
        cd_bg = pred.contiguous().view(-1)
        td_bg = target.contiguous().view(-1)
        intersection = (cd_bg * td_bg).sum()
        A_sum = torch.sum(cd_bg)
        B_sum = torch.sum(td_bg)
        return (2.0 * intersection) / (A_sum + B_sum + smooth)


class cross_entropy_softtargets(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, pred, soft_targets):
        logsoftmax = nn.LogSoftmax()
        return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def basename(arg):
    """
    Obtain basename of a file without the extension
    :param arg: path+filename+extension
    :return: filename
    """
    try:
        return os.path.splitext(os.path.basename(arg))[0]
    except Exception as e:
        if isinstance(arg, list):
            return [basename(el) for el in arg]
        else:
            raise e


def loadImage(fname):
    """
    Load MHD files into a numpy array. Order of axes: [X Y Z]
    :param fname: name of the mhd file (path+filename+extension)
    :return: array containing the image, spacing of the image, offset of the image
    """
    img = sitk.ReadImage(fname)
    if "SegThor" in fname:
        spacing = img.GetSpacing()
        offset = img.GetOrigin()
    else:
        spacing = img.GetSpacing()[::-1]
        offset = img.GetOrigin()[::-1]
    return sitk.GetArrayFromImage(img), spacing, offset


def loadImagewithDir(fname):
    """
    Load MHD files into a numpy array. Order of axes: [X Y Z]
    :param fname: name of the mhd file (path+filename+extension)
    :return: array containing the image, spacing of the image, offset of the image, direction of the image
    """
    img = sitk.ReadImage(fname)
    spacing = img.GetSpacing()[::-1]
    offset = img.GetOrigin()[::-1]
    direction = img.GetDirection()
    return sitk.GetArrayFromImage(img), spacing, offset, direction


def saveImage(fname, arr, spacing=None, direction=None, dtype=np.float32):
    """
    Save a array as MHD file
    :param fname: name of the file to be saved (path + filename + extension
    :param arr: array to be saved
    :param spacing: list defining spacing that the mhd-file needs to have, when spacing==None, spacing==(1,1,1)
    :param dtype: type of stored data
    :return:
    """
    if type(spacing) == type(None):
        spacing = np.ones((len(arr.shape),))
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing(spacing[::-1])
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, fname, True)


def crop_images_to_area_of_interest(image, label, soft_probs, soft_vars, pad_size, ss):
    """Reduce the images to areas where the labels are non-zero

    :param image:
    :param label:
    :param soft_probs:
    :param pad_size:
    :return:
    """
    # first pad both the label and the image to make searching for
    # the bounding box and cropping of the images easier:
    label_padded = np.pad(label, pad_size + ss + 40, mode="constant")
    image_padded = np.pad(image, pad_size + ss + 40, mode="constant")

    assert label_padded.shape == image_padded.shape

    # Find out where we have non-zero labels. Do this for each of the axes separately
    x_non_zero = np.any(label_padded, axis=(1, 2))
    y_non_zero = np.any(label_padded, axis=(0, 2))
    z_non_zero = np.any(label_padded, axis=(0, 1))

    # find the indices where we have non-zero values, and use
    # the first index as the minimum index, and the last (element -1)
    # for the maximum index
    xmin, xmax = np.where(x_non_zero)[0][[0, -1]]
    ymin, ymax = np.where(y_non_zero)[0][[0, -1]]
    zmin, zmax = np.where(z_non_zero)[0][[0, -1]]

    diff_x = xmax - xmin
    diff_y = ymax - ymin
    diff_z = zmax - zmin
    # make the label view a little larger. That way
    # the network will also see the borders of the
    # kidney / tumor
    ss += 8
    if diff_x < ss:
        missing_per_side = (ss - diff_x + 1) // 2
        xmin -= missing_per_side
        xmax += missing_per_side
    if diff_y < ss:
        missing_per_side = (ss - diff_y + 1) // 2
        ymin -= missing_per_side
        ymax += missing_per_side
    if diff_z < ss:
        missing_per_side = (ss - diff_z + 1) // 2
        zmin -= missing_per_side
        zmax += missing_per_side

    xmin -= 20  # make sure bounding box is larger than only the kidneys
    ymin -= 20
    zmin -= 20
    xmax += 20
    ymax += 20
    zmax += 20
    # The labels and soft_probs can simply be cropped exactly to
    # the range of the axes. The main image should be big enough
    # to include the pad_size:
    label = (label_padded[xmin : xmax + 1, ymin : ymax + 1, zmin : zmax + 1]).copy()
    if len(soft_probs) > 0:
        for j in range(len(soft_probs)):
            if j == 0:  # background
                pad_value = 1  # soft probabilities for background should be high
            else:
                pad_value = 0  # soft probabilities for foreground should be low
            soft_prob = soft_probs[j]
            soft_prob_padded = np.pad(
                soft_prob,
                pad_size + ss - 8 + 40,
                mode="constant",
                constant_values=pad_value,
            )
            assert soft_prob_padded.shape == label_padded.shape
            soft_prob = (
                soft_prob_padded[xmin : xmax + 1, ymin : ymax + 1, zmin : zmax + 1]
            ).copy()
            soft_probs[j] = soft_prob
            del soft_prob_padded

    if len(soft_vars) > 0:
        for j in range(len(soft_vars)):
            if j == 0:  # background
                pad_value = 1  # soft probabilities for background should be high
            else:
                pad_value = 0  # soft probabilities for foreground should be low
            soft_var = soft_vars[j]
            soft_var_padded = np.pad(
                soft_var,
                pad_size + ss - 8 + 40,
                mode="constant",
                constant_values=pad_value,
            )
            assert soft_var_padded.shape == label_padded.shape
            soft_var = (
                soft_var_padded[xmin : xmax + 1, ymin : ymax + 1, zmin : zmax + 1]
            ).copy()
            soft_vars[j] = soft_var
            del soft_var_padded

    new_xmin = xmin - pad_size
    new_xmax = xmax + pad_size
    new_ymin = ymin - pad_size
    new_ymax = ymax + pad_size
    new_zmin = zmin - pad_size
    new_zmax = zmax + pad_size
    image = (
        image_padded[
            new_xmin : new_xmax + 1, new_ymin : new_ymax + 1, new_zmin : new_zmax + 1
        ]
    ).copy()

    # Free up numpy related memory. This only worked because we
    # explicitly copied the data that we needed out of the padded
    # block. Because we made a copy, the entire padded data can be
    # deleted now. (There are no references left)
    del label_padded
    del image_padded

    return image, label, soft_probs, soft_vars


def return_center_image(padded_im, orig_im):
    """
    This function returns an unpadded image
    :param padded_im: padded image
    :param orig_im: original image
    :return: Returns center of padded_im according to how much it was padded
    """

    z_padb = max(0, (padded_im.shape[0] - orig_im.shape[0]) // 2)
    y_padb = max(0, (padded_im.shape[1] - orig_im.shape[1]) // 2)
    x_padb = max(0, (padded_im.shape[2] - orig_im.shape[2]) // 2)
    z_pade = -z_padb
    y_pade = -y_padb
    x_pade = -x_padb

    if z_padb == 0:
        z_pade = padded_im.shape[0]
    if y_padb == 0:
        y_pade = padded_im.shape[1]
    if x_padb == 0:
        x_pade = padded_im.shape[2]

    center_im = padded_im[z_padb:z_pade, y_padb:y_pade, x_padb:x_pade]
    return center_im


def obtain_ACDC_scannr(file, ED=True):
    f = open(file)
    number = "XX"
    for x in f:
        if ED and "ED: " in x:
            return x[3:].strip()
        elif not ED and "ES: " in x:
            return x[3:].strip()

    return number


def loadData(
    data_dir,
    rf,
    ss,
    classes,
    soft,
    uncertainty,
    crop_to_labels_of_interest=False,
    pad_all=False,
    ensemble_form="?",
):
    """
    Returns one numpy array containing the images and one nympy array containing the segmentations
    :param data_dir: Directory where data is located
    :param rf: Receptive field of the network (in voxels)
    :param classes: list with class labels
    :param soft: boolean indicating whether we should also load soft
    probabilities. Currently the filenames we expect to find are hard
    coded. See code below.
    :param crop_to_labels_of_interest: boolean indicating whether we want
    crop all images to the the part of the image that contains the
    labels of interest.
    :return:
    """
    cases = os.listdir(data_dir)
    if "soft_probs" in cases:
        cases.remove("soft_probs")
    if "soft_probs_div" in cases:
        cases.remove("soft_probs_div")
    pad_size = rf // 2  # padding on both sides
    if ensemble_form == "uni":
        ens_ext = ""
    elif ensemble_form == "div":
        ens_ext = "_div"

    images = []
    labels = []
    bns = []
    soft_probs = []
    soft_probs_all = []
    soft_vars_all = []
    uncertainty_vars = []
    ss = ss + 1
    for case in tqdm.tqdm(cases):
        if "SegThor" in data_dir:
            im, _, _ = loadImage(
                data_dir + case + "/" + case + "_rs.nii.gz"
            )  # load image
            if "test" not in data_dir:
                segm, _, _ = loadImage(
                    data_dir + case + "/GT_" + case[-2:] + "_rs.nii.gz"
                )  # load segmentation
            else:
                segm = np.zeros(im.shape)
        elif "KiTS19" in data_dir:
            im, _, _ = loadImage(data_dir + case + "/imaging_rs.nii.gz")  # load image
            if "test" not in data_dir:
                segm, _, _ = loadImage(
                    data_dir + case + "/segmentation_rs.nii.gz"
                )  # load segmentation
            else:
                segm = np.zeros(im.shape)
        elif "ACDC" in data_dir:
            # available_scans = os.listdir(data_dir + case)
            # lowest_scan_nr = obtain_ACDC_scannr(available_scans, low=True)
            Infofile = data_dir + case + r"/" + "Info.cfg"
            lowest_scan_nr = obtain_ACDC_scannr(Infofile, ED=True)
            lowest_scan_nr = str(lowest_scan_nr)
            if len(lowest_scan_nr) == 1:
                lowest_scan_nr = "0" + lowest_scan_nr
            im, _, _ = loadImage(
                data_dir
                + case
                + r"/"
                + case
                + "_frame"
                + str(lowest_scan_nr)
                + ".nii.gz"
            )  # load image
            if "test" not in data_dir:
                segm, _, _ = loadImage(
                    data_dir
                    + case
                    + r"/"
                    + case
                    + "_frame"
                    + str(lowest_scan_nr)
                    + "_gt.nii.gz"
                )  # load segmentation
            else:
                segm = np.zeros(im.shape)
        elif "Brains" in data_dir:
            im, _, _ = loadImage(
                data_dir + case + "/" + case + "_im.nii.gz"
            )  # load image
            if "test" not in data_dir:
                segm, _, _ = loadImage(
                    data_dir + case + "/" + case + ".mhd"
                )  # load segmentation
            else:
                segm = np.zeros(im.shape)
        assert segm.shape == im.shape  # make sure labels hae the same shape as image

        if soft:
            # read in soft probabilities
            # get probabilities for background and other classes
            soft_probs_all = list()
            for c in range(len(classes)):
                if "ACDC" in data_dir or "Brains" in data_dir:
                    im_probs, _, _ = loadImage(
                        # data_dir + case + "/" + case + "_ensemble_mean_" + str(classes[c]) + "_soft_probs.nii.gz")
                        data_dir
                        + "soft_probs"
                        + ens_ext
                        + r"/"
                        + case
                        + "_ensemble_mean_"
                        + str(classes[c])
                        + "_soft_probs.nii.gz"
                    )
                    if im_probs.shape != im.shape:
                        im_probs = return_center_image(im_probs, im)
                else:
                    # im_probs, _, _ = loadImage(data_dir + case + "/" + case + "_ensemble_mean_" + str(classes[c]) + "_soft_probs_ds.nii.gz")
                    im_probs, _, _ = loadImage(
                        data_dir
                        + "soft_probs"
                        + ens_ext
                        + r"/"
                        + case
                        + "_ensemble_mean_"
                        + str(classes[c])
                        + "_soft_probs_ds.nii.gz"
                    )
                assert im_probs.shape == im.shape
                if (
                    im.shape[0] < (ss) or im.shape[1] < (ss) or im.shape[2] < (ss)
                ) and not crop_to_labels_of_interest:
                    # ensure that the image fits the sample size:
                    im_probs = np.pad(im_probs, ss // 2, mode="constant")
                soft_probs_all.append(im_probs)

            if uncertainty:
                soft_vars_all = list()
                for c in range(len(classes)):
                    if "ACDC" in data_dir or "Brains" in data_dir:
                        im_vars, _, _ = loadImage(
                            # data_dir + case + "/" + case + "ensemble_std_" + str(classes[c]) + "_soft_probs.nii.gz")
                            data_dir
                            + "soft_probs"
                            + ens_ext
                            + r"/"
                            + case
                            + "_ensemble_std_"
                            + str(classes[c])
                            + "_soft_probs.nii.gz"
                        )
                        if im_vars.shape != im.shape:
                            im_vars = return_center_image(im_vars, im)
                    else:
                        # im_vars, _, _ = loadImage(
                        #     data_dir + case + "/" + case + "ensemble_std_" + str(classes[c]) + "_soft_probs_ds.nii.gz")
                        im_vars, _, _ = loadImage(
                            data_dir
                            + "soft_probs"
                            + ens_ext
                            + r"/"
                            + case
                            + "_ensemble_std_"
                            + str(classes[c])
                            + "_soft_probs_ds.nii.gz"
                        )
                    assert im_vars.shape == im.shape
                    if (
                        im.shape[0] < (ss) or im.shape[1] < (ss) or im.shape[2] < (ss)
                    ) and not crop_to_labels_of_interest:
                        # ensure that the image fits the sample size:
                        im_vars = np.pad(im_vars, ss // 2, mode="constant")
                    soft_vars_all.append(im_vars)

        if crop_to_labels_of_interest:
            im, segm, soft_probs_all, soft_vars_all = crop_images_to_area_of_interest(
                im, segm, soft_probs_all, soft_vars_all, pad_size, ss
            )
        else:
            # ensure that the image fits the sample size:
            if im.shape[0] < (ss) or im.shape[1] < (ss) or im.shape[2] < (ss):
                print(
                    "At least one of the dimensions of this "
                    "input file is too small to fit the sample"
                    "size area. Image: %s, image shape: %s, "
                    "sample size: %s"
                    % (
                        str(data_dir + case + "/imaging_rs.nii.gz"),
                        str(im.shape),
                        str(ss),
                    )
                )
                im = np.pad(im, ss // 2, mode="constant")
                segm = np.pad(segm, ss // 2, mode="constant")
                # im_vars and im_probs are already padded if necessary
            im = np.pad(im, pad_size, mode="constant")

            if pad_all:
                segm = np.pad(segm, pad_size, mode="constant")
                for c in range(len(classes)):
                    if soft_probs_all:  # check if list is not empty
                        soft_probs_all[c] = np.pad(
                            soft_probs_all[c], pad_size, mode="constant"
                        )
                    if soft_vars_all:
                        soft_vars_all[c] = np.pad(
                            soft_vars_all[c], pad_size, mode="constant"
                        )

        images.append(im)
        if soft:
            if pad_all:
                for c in range(len(classes)):
                    assert (
                        soft_probs_all[c].shape == im.shape
                    )  # make sure soft probability maps have same shape as image
            soft_probs.append(soft_probs_all)
        if uncertainty:
            if pad_all:
                for c in range(len(classes)):
                    assert (
                        soft_vars_all[c].shape == im.shape
                    )  # make sure uncertainty maps have same shape as image
            uncertainty_vars.append(soft_vars_all)
        if (
            len(classes) == 2 and 3 in classes
        ):  # make sure labels are correct according to classes
            # 3 in classes means labels should be background-forground [0,1]
            segm[segm > 0] = 1

        labels.append(segm)
        bns.append(case)
        del im, segm
    return (
        np.asarray(images),
        np.asarray(labels),
        bns,
        np.asarray(soft_probs),
        np.asarray(uncertainty_vars),
    )


def loadAllsoftlabels(data_dir, bns, classes):
    labels_bg, _, _ = loadOriginalLabels(data_dir, bns, classes, filename="BG_postprob")
    labels_kidney, _, _ = loadOriginalLabels(
        data_dir, bns, classes, filename="Kidney_postprob"
    )
    if len(classes) == 3:
        labels_tumor, _, _ = loadOriginalLabels(
            data_dir, bns, classes, filename="Tumor_postprob"
        )
    else:
        labels_tumor = list(labels_bg)

    return labels_bg, labels_kidney, labels_tumor


def loadOriginalLabels(data_dir, bns, classes, rf, ss, filename="segmentation"):
    spacings = []
    directions = []
    labels = []
    soft_probs = []
    pad_size = rf // 2
    ss += 1
    for case in tqdm.tqdm(bns):
        if "/" not in filename:
            filename = "/" + filename
        if "segmentation" in filename:
            if "SegThor" in data_dir:
                lab, spacing, _, direction = loadImagewithDir(
                    data_dir + case + "/GT_" + case[-2:] + ".nii.gz"
                )  # load segmentation
            elif "ACDC" in data_dir:
                # available_scans = os.listdir(data_dir + case)
                # lowest_scan_nr = obtain_ACDC_scannr(available_scans, low=True)
                Infofile = data_dir + case + r"/" + "Info.cfg"
                lowest_scan_nr = obtain_ACDC_scannr(Infofile, ED=True)
                lab, spacing, _, direction = loadImagewithDir(
                    data_dir
                    + case
                    + r"/"
                    + case
                    + "_frame"
                    + "0"
                    + str(lowest_scan_nr)
                    + "_gt.nii.gz"
                )  # load segmentation
            elif "Brains" in data_dir:
                lab, spacing, _, direction = loadImagewithDir(
                    data_dir + case + "/" + case + ".mhd"
                )
            else:
                lab, spacing, _, direction = loadImagewithDir(
                    data_dir + case + "/segmentation.nii.gz"
                )  # load segmentation
        else:
            if "SegThor" in data_dir:
                lab, spacing, _, direction = loadImagewithDir(
                    data_dir + case + "/" + case + ".nii.gz"
                )  # load image
            elif "ACDC" in data_dir:
                # available_scans = os.listdir(data_dir + case)
                # lowest_scan_nr = obtain_ACDC_scannr(available_scans, low=True)
                Infofile = data_dir + case + r"/" + "Info.cfg"
                lowest_scan_nr = obtain_ACDC_scannr(Infofile, ED=True)
                lab, spacing, _, direction = loadImagewithDir(
                    data_dir
                    + case
                    + r"/"
                    + case
                    + "_frame"
                    + "0"
                    + str(lowest_scan_nr)
                    + ".nii.gz"
                )  # load image
            else:
                lab, spacing, _, direction = loadImagewithDir(
                    data_dir + case + "/imaging.nii.gz"
                )  # load image

        if (
            len(classes) == 2 and 3 in classes
        ):  # make sure labels are correct according to classes
            # 3 in classes means labels should be background-forground [0,1]
            lab[lab > 0] = 1

        if lab.shape[0] < (ss) or lab.shape[1] < (ss) or lab.shape[2] < (ss):
            print(
                "At least one of the dimensions of this "
                "input file is too small to fit the sample"
                "size area. Image: %s, image shape: %s, "
                "sample size: %s"
                % (str(data_dir + case + "/imaging_rs.nii.gz"), str(lab.shape), str(ss))
            )
            lab = np.pad(lab, ss // 2, mode="constant")

        labels.append(lab)
        spacings.append(spacing)
        directions.append(direction)
    return labels, np.asarray(spacings), np.asarray(directions)


def balanced_batch_iterator2D(
    images,
    labels,
    soft_probs,
    soft_vars,
    soft,
    UP,
    batch_size,
    sample_size,
    classes,
    rf,
    patch_inclusion_criteria="entire-sample-area",
    percent_bg_full_bg=0,
    balance_ratio=[5, 3, 2],
    lossF="CE",
    only_in_plane=False,
    network_arch="dil2D",
):
    """
    Extract 2D patches from the data and put them into a batch
    Batch generation is balanced: 50% is background, 30% is kidney and 20% is tumor

    :param images: Numpy array of all available images
    :param labels: Numpy array of ground truth labels belonging to all available images
    :param batch_size: Size of the batch
    :param sample_size: Size of sample (this is the size of the output of the network)
    :param nclass: Number of classes
    :param rf: Receptive field of the network
    also to label 1 (kidney)
    :return: Batch containing the images and batch with the ground truth segmentations to correspond with the images
    """
    nclass = len(classes)
    assert len(images) == len(labels)
    patch_size = (sample_size + rf, sample_size + rf)  # size of input patches
    if "dil2D" not in network_arch:
        sample_size = (sample_size, sample_size)
    else:
        sample_size = (
            sample_size + 1,
            sample_size + 1,
        )  # for 2d dilated it has to be one voxel bigger (center voxel)
    rs_data = np.random.RandomState(123)
    pad_size = rf // 2

    while True:
        sample_indices = rs_data.randint(len(images), size=batch_size)
        if only_in_plane:
            dirs = np.zeros(batch_size, dtype=int)  # only take slices from Z-direction
        else:
            dirs = rs_data.randint(
                3, size=batch_size
            )  # decide whether it will be an axial, coronal or sagittal slice
        batch = np.empty((batch_size, 1) + patch_size, dtype=np.float32)
        if soft or lossF == "CE":
            targets_labs = np.empty((batch_size,) + sample_size, dtype=np.float32)
        else:
            targets_labs = np.empty(
                (batch_size, nclass) + sample_size, dtype=np.float32
            )
        targets_soft = np.empty((batch_size, nclass) + sample_size, dtype=np.float32)
        targets_vars = np.empty((batch_size, nclass) + sample_size, dtype=np.float32)

        for idx in range(len(sample_indices)):
            # decide of which class the example will be
            required_class_idx = np.random.randint(np.sum(balance_ratio))
            bg_patch_is_full_bg_idx = np.random.randint(1, 100)
            bg_patch_must_be_full_bg = 0
            if nclass > 2:
                if required_class_idx < balance_ratio[0]:
                    required_class = 0  # background
                    bg_patch_must_be_full_bg = (
                        bg_patch_is_full_bg_idx < percent_bg_full_bg
                    )
                else:
                    for c in range(1, nclass):
                        if required_class_idx >= np.sum(
                            balance_ratio[0:c]
                        ) and required_class_idx < np.sum(balance_ratio[0 : c + 1]):
                            required_class = c
            elif nclass == 2:
                if required_class_idx < balance_ratio[0]:
                    required_class = classes[0]
                    bg_patch_must_be_full_bg = (
                        bg_patch_is_full_bg_idx < percent_bg_full_bg
                    )
                else:
                    required_class = classes[1]

            suitable_patch_found = False

            while not suitable_patch_found:  # as long as we have not found a good patch
                n = sample_indices[idx]
                im = images[n]
                lab = labels[n]

                if soft:
                    soft_prob = soft_probs[n]  # either 2 or 3 soft probability maps
                if UP:
                    soft_var_all = soft_vars[n]

                # Extract axial, coronal or sagittal slice
                dir = dirs[idx]
                d = rs_data.randint(lab.shape[dir])
                if dir == 0:
                    im = im[d + pad_size]
                    lab = lab[d]
                    if soft:
                        soft_prob = [item[d] for item in soft_prob]
                        soft_prob = np.asarray(soft_prob)
                    if UP:
                        soft_var_all = [item[d] for item in soft_var_all]
                        soft_var_all = np.asarray(soft_var_all)
                elif dir == 1:
                    im = im[:, d + pad_size, :]
                    lab = lab[:, d, :]
                    if soft:
                        soft_prob = [item[:, d, :] for item in soft_prob]
                        soft_prob = np.asarray(soft_prob)
                    if UP:
                        soft_var_all = [item[:, d, :] for item in soft_var_all]
                        soft_var_all = np.asarray(soft_var_all)
                elif dir == 2:
                    im = im[:, :, d + pad_size]
                    lab = lab[:, :, d]
                    if soft:
                        soft_prob = [item[:, :, d] for item in soft_prob]
                        soft_prob = np.asarray(soft_prob)
                    if UP:
                        soft_var_all = [item[:, :, d] for item in soft_var_all]
                        soft_var_all = np.asarray(soft_var_all)
                else:
                    print("something went wrong while extracting a slice")
                    quit(1)

                offy = np.random.randint(0, im.shape[0] - patch_size[0])
                offx = np.random.randint(0, im.shape[1] - patch_size[1])

                # only accept this patch if the chosen class is in there. If not, grab a new one.
                lab = lab[offy : offy + sample_size[0], offx : offx + sample_size[1]]
                if soft:
                    soft_prob = soft_prob[
                        :, offy : offy + sample_size[0], offx : offx + sample_size[1]
                    ]
                if UP:
                    soft_var_all = soft_var_all[
                        :, offy : offy + sample_size[0], offx : offx + sample_size[1]
                    ]

                sum_required_class_voxels = np.sum(lab == required_class)
                if patch_inclusion_criteria == "entire-sample-area":
                    required_class_in_patch = sum_required_class_voxels > 0
                else:  # 'patch_center_voxel'
                    midpoint_0 = lab.shape[0] // 2
                    midpoint_1 = lab.shape[1] // 2
                    required_class_in_patch = (
                        lab[midpoint_0, midpoint_1] == required_class
                    )

                # special background criteria:
                if required_class == 0 and bg_patch_must_be_full_bg:
                    if sum_required_class_voxels != lab.size:
                        required_class_in_patch = False
                if required_class == 0 and not bg_patch_must_be_full_bg:
                    # we want some non-bg voxels as well:
                    if sum_required_class_voxels == lab.size:
                        required_class_in_patch = False

                if required_class_in_patch:  # a suitable patch was found
                    im = im[offy : offy + patch_size[0], offx : offx + patch_size[1]]
                    suitable_patch_found = True

            # rotation
            if soft or UP:
                rotation_angle = np.random.normal(scale=10)
            else:
                rotation_angle = None

            lab = np.pad(lab, pad_size, mode="constant")

            if soft:
                for s in range(len(soft_prob)):
                    temp_soft_prob = np.pad(soft_prob[s], pad_size, mode="constant")
                    assert im.shape == temp_soft_prob.shape
                    temp_soft_prob = DA.random_rotation(
                        image=temp_soft_prob, mask=None, rotation_angle=rotation_angle
                    )
                    if pad_size > 0:
                        temp_soft_prob = temp_soft_prob[
                            pad_size:-pad_size, pad_size:-pad_size
                        ]
                    soft_prob[s] = temp_soft_prob
            if UP:
                for s in range(len(soft_prob)):
                    temp_soft_var_all = np.pad(
                        soft_var_all[s], pad_size, mode="constant"
                    )
                    assert im.shape == temp_soft_var_all.shape
                    temp_soft_var_all = DA.random_rotation(
                        image=temp_soft_var_all,
                        mask=None,
                        rotation_angle=rotation_angle,
                    )
                    if pad_size > 0:
                        temp_soft_var_all = temp_soft_var_all[
                            pad_size:-pad_size, pad_size:-pad_size
                        ]
                    soft_var_all[s] = temp_soft_var_all

            assert im.shape == lab.shape
            im, lab = DA.random_rotation(
                image=im, mask=lab, rotation_angle=rotation_angle
            )
            if pad_size > 0:
                lab = lab[pad_size:-pad_size, pad_size:-pad_size]
            batch[idx] = im

            if lossF == "CE":
                targets_labs[idx] = lab

            c = 0
            for n in classes:
                if UP:
                    targets_vars[idx, c] = soft_var_all[c]
                if soft:
                    targets_soft[idx, c] = soft_prob[c]
                elif lossF == "dice":
                    l = lab == n
                    l.astype(int)
                    targets_labs[idx, c] = l
                c += 1
        batch = torch.from_numpy(batch).to("cuda")
        if lossF == "CE":
            targets_labs = (
                torch.from_numpy(targets_labs).type(torch.LongTensor).to("cuda")
            )
            targets_soft = (
                torch.from_numpy(targets_soft).type(torch.FloatTensor).to("cuda")
            )
            targets_vars = (
                torch.from_numpy(targets_vars).type(torch.FloatTensor).to("cuda")
            )
        else:
            targets_labs = torch.from_numpy(targets_labs).to("cuda")
            targets_soft = torch.from_numpy(targets_soft).to("cuda")
            targets_vars = torch.from_numpy(targets_vars).to("cuda")
        yield batch, targets_labs, targets_soft, targets_vars


def balanced_batch_iterator3D(
    images,
    labels,
    soft_probs,
    soft_vars,
    soft,
    UP,
    batch_size,
    sample_size,
    classes,
    rf,
    patch_inclusion_criteria="entire-sample-area",
    percent_bg_full_bg=0,
    balance_ratio=[5, 3, 2],
    lossF="CE",
    rotation=False,
):
    """
    Extract 3D patches from the data and put them into a batch
    Batch generation is balanced: 50% is background, 30% is kidney and 20% is tumor

    :param images: Numpy array of all available images
    :param labels: Numpy array of ground truth labels belonging to all available images
    :param batch_size: Size of the batch
    :param sample_size: Size of sample (this is the size of the output of the network)
    :param nclass: Number of classes
    :param rf: Receptive field of the network
    :return: Batch containing the images and batch with the ground truth segmentations to correspond with the images
    """
    nclass = len(classes)
    assert len(images) == len(labels)
    patch_size = (
        sample_size + rf,
        sample_size + rf,
        sample_size + rf,
    )  # size of input patches
    sample_size = (sample_size, sample_size, sample_size)
    rs_data = np.random.RandomState(123)
    pad_size = rf // 2

    while True:
        sample_indices = rs_data.randint(len(images), size=batch_size)
        batch = np.empty((batch_size, 1) + patch_size, dtype=np.float32)
        if soft or lossF == "CE":
            targets_labs = np.empty((batch_size,) + sample_size, dtype=np.float32)
        else:
            targets_labs = np.empty(
                (batch_size, nclass) + sample_size, dtype=np.float32
            )
        targets_soft = np.empty((batch_size, nclass) + sample_size, dtype=np.float32)
        targets_vars = np.empty((batch_size, nclass) + sample_size, dtype=np.float32)

        for idx in range(len(sample_indices)):
            # decide of which class the example will be
            required_class_idx = np.random.randint(np.sum(balance_ratio))
            bg_patch_is_full_bg_idx = np.random.randint(1, 100)
            bg_patch_must_be_full_bg = 0
            if nclass > 2:
                if required_class_idx < balance_ratio[0]:
                    required_class = 0  # background
                    bg_patch_must_be_full_bg = (
                        bg_patch_is_full_bg_idx < percent_bg_full_bg
                    )
                else:
                    for c in range(1, nclass):
                        if required_class_idx >= np.sum(
                            balance_ratio[0:c]
                        ) and required_class_idx < np.sum(balance_ratio[0 : c + 1]):
                            required_class = c
            elif nclass == 2:
                if required_class_idx < balance_ratio[0]:
                    required_class = classes[0]
                    bg_patch_must_be_full_bg = (
                        bg_patch_is_full_bg_idx < percent_bg_full_bg
                    )
                else:
                    required_class = classes[1]

            suitable_patch_found = False

            while not suitable_patch_found:  # as long as we have not found a good patch
                n = sample_indices[idx]
                im = images[n]
                lab = labels[n]
                if soft:
                    soft_prob_all = soft_probs[n]  # either 2 or 3 soft probability maps
                if UP:
                    soft_var_all = soft_vars[n]

                offz = np.random.randint(0, im.shape[0] - patch_size[0])
                offy = np.random.randint(0, im.shape[1] - patch_size[1])
                offx = np.random.randint(0, im.shape[2] - patch_size[2])

                # only accept this patch if the chosen class is in there. If not, grab a new one.
                # CHANGED: for lab and soft_prob used to be sample_size instead of patch_size
                # But the output of resnet and Unet are the same, hence we dont need the sample size!

                lab = lab[
                    offz : offz + sample_size[0],
                    offy : offy + sample_size[1],
                    offx : offx + sample_size[2],
                ]

                if soft:
                    soft_prob = list()
                    for soft_idx in range(len(classes)):
                        soft_prob_t = soft_prob_all[soft_idx][
                            offz : offz + sample_size[0],
                            offy : offy + sample_size[1],
                            offx : offx + sample_size[2],
                        ]
                        soft_prob.append(soft_prob_t)
                if UP:
                    soft_var = list()
                    for soft_idx in range(len(classes)):
                        soft_var_t = soft_var_all[soft_idx][
                            offz : offz + sample_size[0],
                            offy : offy + sample_size[1],
                            offx : offx + sample_size[2],
                        ]
                        soft_var.append(soft_var_t)

                sum_required_class_voxels = np.sum(lab == required_class)
                if patch_inclusion_criteria == "entire-sample-area":
                    required_class_in_patch = sum_required_class_voxels > 0
                else:  # 'patch_center_voxel'
                    midpoint_0 = lab.shape[0] // 2
                    midpoint_1 = lab.shape[1] // 2
                    midpoint_2 = lab.shape[2] // 2
                    required_class_in_patch = (
                        lab[midpoint_0, midpoint_1, midpoint_2] == required_class
                    )

                # special background criteria:
                if required_class == 0 and bg_patch_must_be_full_bg:
                    if sum_required_class_voxels != lab.size:
                        required_class_in_patch = False
                if required_class == 0 and not bg_patch_must_be_full_bg:
                    # we want some non-bg voxels as well:
                    if sum_required_class_voxels == lab.size:
                        required_class_in_patch = False

                if required_class_in_patch:  # a suitable patch was found
                    im = im[
                        offz : offz + patch_size[0],
                        offy : offy + patch_size[1],
                        offx : offx + patch_size[2],
                    ]
                    suitable_patch_found = True

            # rotation
            if rotation:
                if soft:
                    # keep the rotation angle fixed for both the
                    # soft targets, as well as the labels
                    rotation_angle = np.random.normal(scale=10)
                else:
                    rotation_angle = None

                if soft:
                    for s in range(len(soft_prob)):
                        temp_soft_prob = soft_prob[s]
                        # pad the soft probs:
                        temp_soft_prob = np.pad(
                            temp_soft_prob, pad_size, mode="constant"
                        )
                        assert im.shape == temp_soft_prob.shape
                        temp_soft_prob = DA.random_rotation(
                            image=temp_soft_prob,
                            mask=None,
                            rotation_angle=rotation_angle,
                        )
                        temp_soft_prob = temp_soft_prob[
                            pad_size : temp_soft_prob.shape[0] - pad_size,
                            pad_size : temp_soft_prob.shape[1] - pad_size,
                            pad_size : temp_soft_prob.shape[2] - pad_size,
                        ]
                        soft_prob[s] = temp_soft_prob
                if UP:
                    for s in range(len(soft_var)):
                        temp_soft_var = soft_var[s]
                        # pad the soft probs:
                        temp_soft_var = np.pad(temp_soft_var, pad_size, mode="constant")
                        assert im.shape == temp_soft_var.shape
                        temp_soft_var = DA.random_rotation(
                            image=temp_soft_var,
                            mask=None,
                            rotation_angle=rotation_angle,
                        )
                        temp_soft_var = temp_soft_var[
                            pad_size : temp_soft_var.shape[0] - pad_size,
                            pad_size : temp_soft_var.shape[1] - pad_size,
                            pad_size : temp_soft_var.shape[2] - pad_size,
                        ]
                        soft_var[s] = temp_soft_var

                # padd the label file (necessary for the Unet that has a sample
                # size that's smaller than the receptive field) to ensure that
                # to rotations for the image and labels happens in the same way.
                lab = np.pad(lab, pad_size, mode="constant")
                assert im.shape == lab.shape
                im, lab = DA.random_rotation(
                    image=im, mask=lab, rotation_angle=rotation_angle
                )
                # unpad
                lab = lab[
                    pad_size : lab.shape[0] - pad_size,
                    pad_size : lab.shape[1] - pad_size,
                    pad_size : lab.shape[2] - pad_size,
                ]

            batch[idx] = im

            if lossF == "CE":
                targets_labs[idx] = lab

            c = 0
            for n in classes:
                if UP:
                    targets_vars[idx, c] = soft_var[c]
                if soft:
                    targets_soft[idx, c] = soft_prob[c]
                elif lossF == "dice":
                    l = lab == n
                    l.astype(int)
                    targets_labs[idx, c] = l
                c += 1

        batch = torch.from_numpy(batch).to("cuda")
        if lossF == "CE":
            targets_labs = (
                torch.from_numpy(targets_labs).type(torch.LongTensor).to("cuda")
            )
            targets_soft = (
                torch.from_numpy(targets_soft).type(torch.FloatTensor).to("cuda")
            )
            targets_vars = (
                torch.from_numpy(targets_vars).type(torch.FloatTensor).to("cuda")
            )
        else:
            targets_labs = torch.from_numpy(targets_labs).to("cuda")
            targets_soft = torch.from_numpy(targets_soft).to("cuda")
            targets_vars = torch.from_numpy(targets_vars).to("cuda")
        yield batch, targets_labs, targets_soft, targets_vars


def makeBB(image, labels):
    # make a binary image in which bounding boxes are present
    bb_im = np.zeros(image.shape)
    for i in range(1, labels + 1):
        coords = np.where(image == i)
        bb_im[
            np.min(coords[0]) : np.max(coords[0]) + 1,
            np.min(coords[1]) : np.max(coords[1]) + 1,
            np.min(coords[2]) : np.max(coords[2]) + 1,
        ] = 1
    return bb_im
