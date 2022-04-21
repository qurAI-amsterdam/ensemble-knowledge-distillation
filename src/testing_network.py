import argparse
import os
import sys
import time

import numpy as np
import pandas
import torch
import tqdm
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from skimage import measure

import networks as nw
import resampling_norounding as rn
import utils as u


def retain_largest_components(
    mask, labels=None, n=1, background=0, connectivity=None, label_indiv=False
):
    """
    From Nikolas
    Returns a new numpy array with only the n largest connected components retained per label.

    Parameters
    ----------
    mask : ndarray
        Numpy array with integer labels

    labels : iterable or None
        List of labels to retain. If not provided, the mask is first searched for unique values.

    n : int
        Number of components to retain per label

    background : int
        Background value

    connectivity : int or None
        Determines the connectivity that defines connected-ness. Values between 1 and mask.ndim are
        permitted, see manual of skimage.measure.label for details. Defaults to full connectivity.

    Returns
    -------
    ndarray
        Numpy array with the same shape and dtype as the input mask
    """
    # Determine labels present in the mask if a list of labels was not provided
    if labels is None:
        labels = np.unique(mask[mask != background])

    reduced_mask = np.full_like(mask, fill_value=background)
    for label in labels:
        cmap = measure.label(
            mask == label, background=background, connectivity=connectivity
        )
        components = np.unique(cmap[cmap > 0], return_counts=True)

        for i, component in enumerate(
            sorted(zip(*components), key=lambda c: c[1], reverse=True)
        ):
            if i == n:
                break
            if label_indiv:
                reduced_mask[cmap == component[0]] = i + 1
            else:
                reduced_mask[cmap == component[0]] = label

    return reduced_mask


def dice_score(mask1, mask2):
    """From Nikolas: Dice volume overlap score for two binary masks"""
    m1 = np.asarray(mask1, dtype="bool").flatten()
    m2 = np.asarray(mask2, dtype="bool").flatten()

    try:
        return (
            2
            * np.count_nonzero(m1 & m2)
            / float(np.count_nonzero(m1) + np.count_nonzero(m2))
        )
    except ZeroDivisionError:
        raise ValueError("Cannot compute dice score on empty masks")


def surface_distances(manual, automatic, voxel_spacing=None, connectivity=None):
    """Computes the surface distances (positive numbers) from all border voxels of a binary object in two images.
    http://nikolas.ds.umcutrecht.nl/qiapython/docs/modules/metrics.html"""
    manual = np.asarray(manual, dtype="bool")
    automatic = np.asarray(automatic, dtype="bool")

    if np.count_nonzero(manual) == 0 or np.count_nonzero(automatic) == 0:
        raise ValueError(
            "Cannot compute surface distance if there are no foreground pixels in the image"
        )

    if connectivity is None:
        connectivity = manual.ndim

    # Extract border using erosion
    footprint = generate_binary_structure(manual.ndim, connectivity)
    manual_border = manual ^ binary_erosion(manual, structure=footprint, iterations=1)
    automatic_border = automatic ^ binary_erosion(
        automatic, structure=footprint, iterations=1
    )

    # Compute average surface distance
    dt = distance_transform_edt(~manual_border, sampling=voxel_spacing)
    return dt[automatic_border]


def hausdorff_distance(
    manual,
    automatic,
    voxel_spacing=None,
    connectivity=None,
    symmetric=True,
    percentile=False,
):
    """
    http://nikolas.ds.umcutrecht.nl/qiapython/docs/modules/metrics.html
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two images.

    Parameters
    ----------
    manual : numpy array
        Reference masks (binary)

    automatic : numpy array
        Masks that is compared to the reference mask

    voxel_spacing : None or sequence of floats
        Spacing between elements in the images

    connectivity : int
        The neighbourhood/connectivity considered when determining the surface of the binary objects. Values between 1 and ndim are valid.
        Defaults to ndim, which is full connectivity even along the diagonal.

    symmetric : bool
        Whether the distance is calculated from manual to automatic mask, or symmetrically (max distance in either direction)

    Returns
    -------
    float
        Hausdorff distance
    """
    # hd1 = surface_distances(manual, automatic, voxel_spacing, connectivity).max()
    hd1 = surface_distances(manual, automatic, voxel_spacing, connectivity)
    if not symmetric:
        if not percentile:  # just return max and not 95th percentile
            return hd1.max()
        return np.percentile(hd1, 95)

    # hd2 = surface_distances(automatic, manual, voxel_spacing, connectivity).max()
    hd2 = surface_distances(automatic, manual, voxel_spacing, connectivity)
    if not percentile:
        return max(hd1.max(), hd2.max())
    return max(np.percentile(hd1, 95), np.percentile(hd2, 95))


def average_surface_distance(
    manual, automatic, voxel_spacing=None, connectivity=None, symmetric=True
):
    """
    http://nikolas.ds.umcutrecht.nl/qiapython/docs/modules/metrics.html
    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    manual : numpy array
        Reference masks (binary)

    automatic : numpy array
        Masks that is compared to the reference mask

    voxel_spacing : None or sequence of floats
        Spacing between elements in the images

    connectivity : int
        The neighbourhood/connectivity considered when determining the surface of the binary objects. Values between 1 and ndim are valid.
        Defaults to ndim, which is full connectivity even along the diagonal.

    symmetric : bool
        Whether the surface distance are calculated from manual to automatic mask, or symmetrically in both directions

    Returns
    -------
    float
        Average surface distance
    """
    sd1 = surface_distances(manual, automatic, voxel_spacing, connectivity)
    if not symmetric:
        return sd1.mean()

    sd2 = surface_distances(automatic, manual, voxel_spacing, connectivity)
    return np.concatenate((sd1, sd2)).mean()


def Network2D(
    data_dir,
    save_dir,
    run,
    eval,
    network_weights,
    receptive_field,
    sample_size,
    classes,
    clip,
    temperature,
    crop_image_to_non_zero_labels,
    return_soft_probs=False,
    save_soft_probs=True,
    save_predictions=True,
    return_only_ds_soft_probs=False,
    return_filenames_of_softprobs=False,
    network_arch="dil2D",
    create_mask=False,
    predict_uncertainty=False,
    seed=-1,
):
    """
    Evaluation of 2D network

    :param data_dir: Directory where data is stored
    :param save_dir: Directory where results stored
    :param run: Name of the experiment
    :param eval: Evaluation data--> 'val', 'test' or 'train'
    :return:
    """
    if return_soft_probs and return_filenames_of_softprobs:
        print(
            "Error: the current implementation only facilitates returning "
            "either the soft probabilities or the filenames of the saved "
            "soft probabilities. "
        )
        raise ValueError

    if return_soft_probs:
        soft_probs = []

    if return_filenames_of_softprobs:
        filenames_soft_probs = []
        if not save_soft_probs:
            print(
                "Warning: you specified that you want to have the "
                "filenames of the soft probabilities returned, but "
                "not that you want to save the soft probabilities. This "
                "needs to be done, so changing this setting now."
            )
            save_soft_probs = True

    w = network_weights
    run_name = os.path.splitext(os.path.basename(w))[0]
    loss_func = run[run.find("loss") + 5 : run.find("loss") + 9]

    classes = classes.split(",")
    classes = np.asarray(classes, dtype=int)
    nclass = len(classes)

    # if not (receptive_field == 131 or receptive_field == 67 or receptive_field == 35):
    #     print("Unknown (or unimplemented) receptive field size provided: %s" % str(receptive_field))
    #     sys.exit()

    rf = receptive_field
    ss = sample_size  # sample size
    pw = rf // 2  # pad-width
    # patch_size_net = receptive_field + sample_size
    patch_size_net = 8

    # create the output save directory if it does not exist:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # network architecture
    if network_arch == "Unet2DSmallOrig":
        network = nw.UNet2DSmallOrig(
            nclass, temperature=temperature, deep_supervision=False
        )
    elif network_arch == "Resnet2DNEW":
        network = nw.Resnet2D(
            nclass, temperature=temperature, architecture=network_arch
        )
    elif network_arch == "dil2D":
        network = nw.DilatedNetwork2D(
            nclass,
            receptive_field=rf,
            temperature=temperature,
            uncertainty_prediction=predict_uncertainty,
        )
    else:
        print("unknown architecture")
        sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.load_state_dict(torch.load(w))
    network.eval()  # set dropout and batch normalization layers to evaluation mode
    torch.no_grad()
    nr_parameters = network.count_parameters(trainable=True)

    # images used to be padded with ss=ss
    test_images, test_labels_ds, bns, _, _ = u.loadData(
        data_dir + "/" + eval + "/",
        rf=rf,
        ss=ss,
        classes=classes,
        soft=False,
        uncertainty=False,
        crop_to_labels_of_interest=crop_image_to_non_zero_labels,
        pad_all=(network_arch != "dil2D"),
    )  # Load evaluation data
    if create_mask:
        file_name = ""  # we dont have the ground truth segmentations
    elif "test" in eval:
        if "SegThor" in data_dir or "Brains" in data_dir:
            file_name = "segmentation"  # we have the ground truth segmentations
        else:
            file_name = ""  # we dont have the ground truth segmentations
    else:
        file_name = "segmentation"

    test_labels, spacings_orig, directions_orig = u.loadOriginalLabels(
        data_dir + "/" + eval + "/",
        bns,
        classes=classes,
        rf=rf,
        ss=ss,
        filename=file_name,
    )  # obtain spacing of original images
    if "KiTS" in data_dir:
        spacing = (1.5, 1.5, 2.5)  # spacing of analyzed images
    elif "SegThor" in data_dir:
        spacing = (
            2.5,
            0.9765620231628418,
            0.9765620231628418,
        )  # spacing of analyzed images
    # make sure labels are valid when network is trained for background vs kidney+tumor

    if nclass == 2 and 3 in classes:  # binarize ground truth
        idx = np.where(classes == 3)
        classes[idx] = 1

    Dice = np.empty((len(test_images), nclass))  # array to store dices in per class
    ASD = np.empty((len(test_images), nclass))  # array to store dices in per class
    HD = np.empty((len(test_images), nclass))  # array to store dices in per class
    HD_p = np.empty((len(test_images), nclass))  # array to store dices in per class
    times = []
    for imidx in range(len(test_images)):
        bn = bns[imidx]
        print("Analyzing image: ", bn)

        im = test_images[imidx]
        im = np.asarray(im, dtype=np.float64)

        if network_arch != "dil2D":
            # make sure image fits in receptive field
            pad_z = patch_size_net - im.shape[0] % patch_size_net
            pad_y = patch_size_net - im.shape[1] % patch_size_net
            pad_x = patch_size_net - im.shape[2] % patch_size_net
            im = np.pad(im, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")

        im = torch.from_numpy(im).type(torch.FloatTensor).cuda()
        if return_only_ds_soft_probs:
            lab = test_labels_ds[imidx]
        else:
            lab = test_labels[imidx]
        if create_mask:
            lab_ds = test_labels_ds[imidx]

        if "ACDC" in data_dir or "Brains" in data_dir:
            spacing = spacings_orig[imidx]  # these images are not resampled
        spacing_orig = spacings_orig[imidx]
        direction_orig = directions_orig[imidx]

        # image in which total result is stored
        if network_arch == "dil2D":
            outim = np.empty(
                (
                    nclass,
                    im.shape[0] - rf + 1,
                    im.shape[1] - rf + 1,
                    im.shape[2] - rf + 1,
                )
            )
            if predict_uncertainty:
                outim_UP = np.empty(
                    (
                        nclass,
                        im.shape[0] - rf + 1,
                        im.shape[1] - rf + 1,
                        im.shape[2] - rf + 1,
                    )
                )
        else:
            outim = np.empty(
                (nclass, im.shape[0] - rf, im.shape[1] - rf, im.shape[2] - rf)
            )
            if predict_uncertainty:
                outim_UP = np.empty(
                    (nclass, im.shape[0] - rf, im.shape[1] - rf, im.shape[2] - rf)
                )

        # Evaluation of axial slices
        print("Evaluating Axial slices...")
        start = time.time()
        for im_slice in tqdm.tqdm(range(pw, im.shape[0] - pw - 1)):
            im_in = im[im_slice]
            im_in = im_in.reshape((1, 1, im_in.shape[0], im_in.shape[1]))
            if predict_uncertainty:
                prediction, logits, pred_unc = network(im_in)
                for n in range(nclass):
                    outim[n, im_slice - pw] += prediction[0, n].cpu().detach().numpy()
                    outim_UP[n, im_slice - pw] += pred_unc[0, n].cpu().detach().numpy()
            else:
                prediction, logits = network(im_in)
                for n in range(nclass):
                    outim[n, im_slice - pw] += prediction[0, n].cpu().detach().numpy()
            del im_in

        nr_preds = 1.0
        if (
            "ACDC" not in data_dir
        ):  # all other datasets are analyzed in three directions, ACDC in only one
            nr_preds = 3.0
            # Evaluation of coronal slices
            print("Evaluating Coronal slices...")
            for im_slice in tqdm.tqdm(range(pw, im.shape[1] - pw - 1)):
                im_in = im[:, im_slice, :]
                im_in = im_in.reshape((1, 1, im_in.shape[0], im_in.shape[1]))
                if predict_uncertainty:
                    prediction, logits, pred_unc = network(im_in)
                    outim_UP[n, :, im_slice - pw, :] += (
                        pred_unc[0, n].cpu().detach().numpy()
                    )
                else:
                    prediction, logits = network(im_in)
                for n in range(nclass):
                    outim[n, :, im_slice - pw, :] += (
                        prediction[0, n].cpu().detach().numpy()
                    )
                del im_in

            # Evaluation of sagittal slices
            print("Evaluating Sagittal slices...")
            for im_slice in tqdm.tqdm(range(pw, im.shape[2] - pw - 1)):
                im_in = im[:, :, im_slice]
                im_in = im_in.reshape((1, 1, im_in.shape[0], im_in.shape[1]))
                if predict_uncertainty:
                    prediction, logits, pred_unc = network(im_in)
                    outim_UP[n, :, :, im_slice - pw] += (
                        pred_unc[0, n].cpu().detach().numpy()
                    )
                else:
                    prediction, logits = network(im_in)
                for n in range(nclass):
                    outim[n, :, :, im_slice - pw] += (
                        prediction[0, n].cpu().detach().numpy()
                    )
                del im_in

        print("Computing and saving final result...")
        all_output = []
        all_output_UP = []
        all_output_ds = []
        for n in range(nclass):
            if predict_uncertainty:
                out_UP = np.squeeze(outim_UP[n, :, :, :]) / nr_preds
            out = np.squeeze(outim[n, :, :, :]) / nr_preds
            if network_arch != "dil2D":
                out = out[0:-pad_z, 0:-pad_y, 0:-pad_x]
                if predict_uncertainty:
                    out_UP = out_UP[0:-pad_z, 0:-pad_y, 0:-pad_x]
            if create_mask:  # we need to save downsampled and original spacing images
                out_ds = np.copy(out)
                all_output_ds.append(out_ds)
            if not return_only_ds_soft_probs:
                out = rn.resample_image(out, spacing, spacing_orig, lab.shape, order=1)
                if predict_uncertainty:
                    out_UP = rn.resample_image(
                        out_UP, spacing, spacing_orig, lab.shape, order=1
                    )
            all_output.append(out)
            if predict_uncertainty:
                all_output_UP.append(out_UP)
        if return_soft_probs:
            soft_probs.append(all_output)
        del outim

        # final result image
        out_im_class = np.zeros(
            (lab.shape[0], lab.shape[1], lab.shape[2]), dtype="float32"
        )
        for z in range(lab.shape[0]):
            outtmp = np.zeros((nclass, lab.shape[1], lab.shape[2]))
            for n in range(nclass):
                outtmp[n, :, :] = np.squeeze(all_output[n][z])
            out_im_class[z] = np.squeeze(np.argmax(outtmp, axis=0))
            del outtmp
        end = time.time()
        times.append(end - start)

        if predict_uncertainty:
            for n in range(nclass):
                if "test" in eval or "ACDC" in data_dir:
                    u.saveImage(
                        save_dir + "/" + bn + "_uncertainty_class" + str(n) + ".nii.gz",
                        all_output_UP[n][
                            ss // 2 : -ss // 2, ss // 2 : -ss // 2, ss // 2 : -ss // 2
                        ],
                        spacing=spacing_orig,
                        direction=direction_orig,
                    )

        if create_mask:
            out_im_class_ds = np.zeros(
                (lab_ds.shape[0], lab_ds.shape[1], lab_ds.shape[2]), dtype="float32"
            )
            for z in range(lab_ds.shape[0]):
                outtmp_ds = np.zeros((nclass, lab_ds.shape[1], lab_ds.shape[2]))
                for n in range(nclass):
                    outtmp_ds[n, :, :] = np.squeeze(all_output_ds[n][z])
                out_im_class_ds[z] = np.squeeze(np.argmax(outtmp_ds, axis=0))
                del outtmp_ds

        if create_mask:  # dilate outcome with 5 voxels on all sides
            out_im_class = retain_largest_components(out_im_class, n=2)
            out_im_classlab = retain_largest_components(
                out_im_class, n=2, label_indiv=True
            )
            out_im_class = binary_dilation(out_im_class, np.ones((5, 5, 5)))
            bb_im = u.makeBB(out_im_classlab, labels=2)

            out_im_class_ds = retain_largest_components(out_im_class_ds, n=2)
            out_im_classlab_ds = retain_largest_components(
                out_im_class_ds, n=2, label_indiv=True
            )
            out_im_class_ds = binary_dilation(out_im_class_ds, np.ones((5, 5, 5)))
            bb_im_ds = u.makeBB(out_im_classlab_ds, labels=2)

        if clip:  # mask output of network with ground truth segmentation
            lab_bin = lab > 0
            out_im_class = out_im_class * lab_bin

        # save final result image
        im_name = (
            save_dir
            + "/"
            + bn
            + "_prediction_using_"
            + network_arch
            + "_"
            + loss_func
            + "_"
            + str(seed)
        )
        if "ACDC" in data_dir:
            im_name = save_dir + "/" + bn + "_ED"
        if save_predictions and not return_only_ds_soft_probs:
            # u.saveImage(save_dir + "/" + bn + "_prediction_using_" + run_name + ".nii.gz", out_im_class,
            if create_mask:
                u.saveImage(
                    save_dir + "/" + bn + "_mask.nii.gz",
                    out_im_class,
                    spacing=spacing_orig,
                    direction=direction_orig,
                )
                u.saveImage(
                    save_dir + "/" + bn + "_bb.nii.gz",
                    bb_im,
                    spacing=spacing_orig,
                    direction=direction_orig,
                )
            else:
                if "test" in eval and "ACDC" in data_dir:
                    out_im_class = out_im_class[
                        ss // 2 : -ss // 2, ss // 2 : -ss // 2, ss // 2 : -ss // 2
                    ]
                    # lab = lab[ss//2:-ss//2, ss//2:-ss//2, ss//2:-ss//2] #when and becomes or for testing on validationset acdc
                # print(spacing_orig, direction_orig)
                u.saveImage(
                    im_name + ".nii.gz",
                    out_im_class,
                    spacing=spacing_orig,
                    direction=direction_orig,
                    dtype="uint8",
                )
        elif save_predictions and return_only_ds_soft_probs:
            u.saveImage(
                im_name + "_ds.nii.gz",
                out_im_class,
                spacing=spacing,
                direction=direction_orig,
            )
        if create_mask:
            u.saveImage(
                save_dir + "/" + bn + "_mask_ds.nii.gz",
                out_im_class_ds,
                spacing=spacing,
                direction=direction_orig,
            )
            u.saveImage(
                save_dir + "/" + bn + "_bb_ds.nii.gz",
                bb_im_ds,
                spacing=spacing,
                direction=direction_orig,
            )

        all_filenames_soft_probs = []
        if return_soft_probs or return_filenames_of_softprobs:
            # if save_soft_probs and not return_only_ds_soft_probs:
            for n in range(nclass):
                soft_prob_filename = (
                    save_dir
                    + "/"
                    + bn
                    + "_soft_probs_"
                    + str(n)
                    + "_using_"
                    + network_arch
                    + "_"
                    + loss_func
                    + "_"
                    + str(seed)
                    + ".nii.gz"
                )
                if return_only_ds_soft_probs:
                    soft_prob_filename = (
                        save_dir
                        + "/"
                        + bn
                        + "_soft_probs_"
                        + str(n)
                        + "_using_"
                        + network_arch
                        + "_"
                        + loss_func
                        + "_"
                        + str(seed)
                        + "_ds.nii.gz"
                    )
                if save_soft_probs and not return_only_ds_soft_probs:
                    u.saveImage(
                        soft_prob_filename,
                        all_output[n],
                        spacing=spacing_orig,
                        direction=direction_orig,
                    )
                elif save_soft_probs and return_only_ds_soft_probs:
                    u.saveImage(
                        soft_prob_filename,
                        all_output[n],
                        spacing=spacing,
                        direction=direction_orig,
                    )
                all_filenames_soft_probs.append(soft_prob_filename)

        if return_filenames_of_softprobs:
            filenames_soft_probs.append(all_filenames_soft_probs)

        # compute_dice = "test" not in eval
        compute_dice = (
            "test" not in eval or "SegThor" in data_dir or "Brains" in data_dir
        )
        if "train" in eval:
            compute_dice = False
        # if not return_only_ds_soft_probs and compute_dice:
        #     i = 0
        #     for n in classes:
        #         av_surf_dist=10000
        #         haus_d = 10000
        #         haus_dp = 10000
        #         pred = out_im_class==n
        #         # pred = retain_largest_components(pred, n = 1)
        #         ref = lab==n
        #         assert (ref.shape == pred.shape)
        #         dice = dice_score(ref, pred)
        #         Dice[imidx, i] = dice
        #         try:
        #             av_surf_dist = average_surface_distance(ref, pred, spacing_orig)
        #             haus_d = hausdorff_distance(ref, pred, spacing_orig)
        #             haus_dp = hausdorff_distance(ref, pred, spacing_orig, percentile=True)
        #         except:
        #             continue
        #         ASD[imidx, i] = av_surf_dist
        #         HD[imidx, i] = haus_d
        #         HD_p[imidx, i] = haus_dp
        #         i += 1
        #         print("Dice for class ", str(n), " is: ", str(dice))
        #         print("Average surface distance for class ", str(n), " is: ", str(av_surf_dist))
        #         print("Hausdorff Distance for class ", str(n), " is: ", str(haus_d))

    if not return_only_ds_soft_probs:
        for n in range(nclass):
            print("Class: ", n)
            print("AVERAGE DICE: ", np.average(Dice[:, n]))
            print("STD DEV DICE: ", np.std(Dice[:, n]))
            print("MAX DICE: ", np.max(Dice[:, n]), np.argmax(Dice[:, n]))
            print("MIN DICE: ", np.min(Dice[:, n]), np.argmin(Dice[:, n]))
            print("")

            # print("Class: ", n)
            # print("AVERAGE ASD: ", np.average(ASD[:, n]))
            # print("STD DEV ASD: ", np.std(ASD[:, n]))
            # print("MAX ASD: ", np.max(ASD[:, n]), np.argmax(ASD[:, n]))
            # print("MIN ASD: ", np.min(ASD[:, n]), np.argmin(ASD[:, n]))
            # print("")
            #
            # print("Class: ", n)
            # print("AVERAGE HD: ", np.average(HD[:, n]))
            # print("STD DEV HD: ", np.std(HD[:, n]))
            # print("MAX HD: ", np.max(HD[:, n]), np.argmax(HD[:, n]))
            # print("MIN HD: ", np.min(HD[:, n]), np.argmin(HD[:, n]))
            # print("")

        df = pandas.DataFrame(Dice)
        df.to_excel(
            save_dir + "/DICE_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )
        df = pandas.DataFrame(bns)
        df.to_excel(save_dir + "/basenames.xlsx", index=False)
        df = pandas.DataFrame(ASD)
        df.to_excel(
            save_dir + "/ASD_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )

        df = pandas.DataFrame(HD)
        df.to_excel(
            save_dir + "/HD_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )
        df = pandas.DataFrame(HD_p)
        df.to_excel(
            save_dir + "/HD95_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )
    df = pandas.DataFrame(times)
    df.to_excel(
        save_dir + "/InferenceTimes_" + network_arch + "_" + loss_func + ".xlsx",
        index=False,
    )

    max_gpu_usage = torch.cuda.max_memory_allocated(device=None)
    file_obj = open(save_dir + "/" + run_name + ".txt", "w")
    file_obj.write(run_name)
    file_obj.write("Max GPU usage: " + str(max_gpu_usage))
    file_obj.write("nr_parameters: " + str(nr_parameters))

    torch.cuda.reset_max_memory_allocated(device=None)

    if return_soft_probs:
        return soft_probs
    elif return_filenames_of_softprobs:
        return filenames_soft_probs
    else:
        return None


def DilatedNetwork3D(data_dir, weight_dir, save_dir, run, network_arch, eval):
    """
    Evaluation of 3D dilated network

    :param data_dir: Directory where data is stored
    :param weight_dir: Directory where weights are stored
    :param save_dir: Directory where results stored
    :param run: Name of the experiment
    :param network_arch: Network architecture
    :param eval: Evaluation data--> 'val', 'test' or 'train'
    :return:
    """
    iter = 60000
    w = weight_dir + "Temp_" + run + network_arch + "_" + str(iter) + ".pth"
    # w = weight_dir + "FINAL_" + run + network_arch + "_" + str(iter) + ".pth"

    rf = 67  # receptive field
    ss = 10  # sample size
    bs = 10  # batch size
    pw = rf // 2  # pad-width

    network = nw.DilatedNetwork3D(nclass)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.load_state_dict(torch.load(w))
    network.eval()  # set dropout and batch normalization layers to evaluation mode
    torch.no_grad()

    test_images, _, bns = u.loadData(
        data_dir + "/" + eval + "/", rf=rf, ss=ss
    )  # Load evaluation data
    test_labels, spacings_orig, directions_orig = u.loadOriginalLabels(
        data_dir + "/" + eval + "/", bns
    )  # obtain spacing of original images
    spacing = (1.5, 1.5, 2.5)  # spacing of analyzed images

    Dice = np.empty((len(test_images), nclass))  # array to store dices in per class
    for imidx in range(len(test_images)):
        bn = bns[imidx]
        print("Analyzing image: ", bn)

        im = test_images[imidx]
        im = torch.from_numpy(im).type(torch.FloatTensor).cuda()
        im = im.reshape((1, 1, im.shape[0], im.shape[1], im.shape[2]))
        lab = test_labels[imidx]
        spacing_orig = spacings_orig[imidx]
        direction_orig = directions_orig[imidx]

        # image in which total result is stored
        outim = np.empty(
            (nclass, im.shape[2] - rf + 1, im.shape[3] - rf + 1, im.shape[4] - rf + 1)
        )
        for z in tqdm.tqdm(range(im.shape[4] - rf)):
            im_in = im[:, :, :, :, z : z + rf]
            prediction = network(im_in).cpu().detach().numpy()
            for n in range(nclass):
                outim[n, :, :, z] += (prediction[0, n, :, :, 0] > 0.5) * n
            del im_in, prediction

        # outim = network(im)

        print("Computing and saving final result...")
        out_bg = np.squeeze(outim[0, :, :, :])  # Background
        out_bg = rn.resample_image(
            out_bg, spacing, spacings_orig, lab.shape, order=1
        )  # resample back to original resolution

        out_kidney = np.squeeze(outim[1, :, :, :])  # Kidney
        out_kidney = rn.resample_image(
            out_kidney, spacing, spacings_orig, lab.shape, order=1
        )  # resample back to original resolution

        out_tumor = np.squeeze(outim[2, :, :, :])  # Tumor
        out_tumor = rn.resample_image(
            out_tumor, spacing, spacings_orig, lab.shape, order=1
        )  # resample back to original resolution
        del outim

        # final result image
        out_im_class = np.zeros(
            (lab.shape[0], lab.shape[1], lab.shape[2]), dtype="float32"
        )
        for z in range(lab.shape[0]):
            outtmp = np.zeros((nclass, lab.shape[1], lab.shape[2]))
            outtmp[0, :, :] = np.squeeze(out_bg[z])
            outtmp[1, :, :] = np.squeeze(out_kidney[z])
            outtmp[2, :, :] = np.squeeze(out_tumor[z])
            out_im_class[z] = np.squeeze(np.argmax(outtmp, axis=0))
            del outtmp
        u.saveImage(
            save_dir + bn + "prediction.nii.gz",
            out_im_class,
            spacing=spacing_orig,
            direction=direction_orig,
        )

        for n in range(nclass):
            pred = out_im_class == n
            # pred = retain_largest_components(pred, n=1)
            ref = lab == n
            dice = dice_score(ref, pred)
            Dice[imidx, n] = dice
            print("Dice for class " + str(n) + " is: " + str(dice))

    for n in range(nclass):
        print("Class: ", n)
        print("AVERAGE DICE: " + np.average(Dice[:, n]))
        print("STD DEV DICE: " + np.std(Dice[:, n]))
        print("MAX DICE: " + np.max(Dice[:, n]) + " Image: " + np.argmax(Dice[:, n]))
        print("MIN DICE: " + np.min(Dice[:, n]) + " Image: " + np.argmin(Dice[:, n]))
        print("")

    df = pandas.DataFrame(Dice)
    df.to_excel(save_dir + run + "DICE.xlsx", index=False)
    df = pandas.DataFrame(bns)
    df.to_excel(save_dir + run + "basenames.xlsx", index=False)


def Network3D(
    data_dir,
    save_dir,
    run,
    eval,
    network_arch,
    network_weights,
    receptive_field,
    sample_size,
    classes,
    clip,
    temperature,
    crop_image_to_non_zero_labels,
    number_of_avgs_for_prediction=8,
    return_soft_probs=False,
    save_soft_probs=True,
    save_predictions=True,
    only_down_sampled_soft_probs=False,
    return_filenames_of_softprobs=False,
    predict_uncertainty=False,
    seed=-1,
):
    if return_soft_probs and return_filenames_of_softprobs:
        print(
            "Error: the current implementation only facilitates returning "
            "either the soft probabilities or the filenames of the saved "
            "soft probabilities. "
        )
        raise ValueError

    if return_soft_probs:
        soft_probs = []

    if return_filenames_of_softprobs:
        filenames_soft_probs = []
        if not save_soft_probs:
            print(
                "Warning: you specified that you want to have the "
                "filenames of the soft probabilities returned, but "
                "not that you want to save the soft probabilities. This "
                "needs to be done, so changing this setting now."
            )
            save_soft_probs = True

    w = network_weights
    run_name = os.path.splitext(os.path.basename(w))[0]
    loss_func = run[run.find("loss") + 5 : run.find("loss") + 9]

    classes = classes.split(",")
    classes = np.asarray(classes, dtype=int)
    nclass = len(classes)

    rf = receptive_field
    ss = sample_size  # sample size
    pw = rf // 2  # pad-width
    patch_size_net = receptive_field + sample_size

    print("Patch-size is: ", patch_size_net)

    # create the output save directory if it does not exist:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if network_arch == "Unet3D":
        network = nw.UNet3D(nclass, temperature=temperature)  # network architecture
    elif network_arch == "Unet3DSmall" or network_arch == "Unet3DSmallELU":
        network = nw.UNet3DSmall(
            nclass, temperature=temperature, use_ELU=(network_arch == "Unet3DSmallELU")
        )
    elif network_arch == "Unet3DSmallDS":
        network = nw.UNet3DSmallDS(nclass, temperature=temperature)
    elif network_arch == "Unet3DSmallOrig" or network_arch == "Unet3DSmallOrigDS":
        network = nw.UNet3DSmallOrig(
            nclass,
            temperature=temperature,
            deep_supervision=(network_arch == "Unet3DSmallOrigDS"),
            uncertainty_prediction=predict_uncertainty,
        )
    elif network_arch == "Unet3DStrided":
        network = nw.UNet3D_stridedConv(
            nclass, temperature=temperature
        )  # network architecture
    elif network_arch == "Unet3DSmall":
        network = nw.UNet3DSmall(
            nclass, temperature=temperature
        )  # network architecture
    elif network_arch == "Resnet3D" or network_arch == "Resnet3DNEW":
        network = nw.Resnet3D(
            nclass, temperature=temperature, architecture=network_arch
        )  # network architecture
    elif network_arch == "Resnet3DUnc":
        network = nw.Resnet3DUncertainty(
            nclass, temperature=temperature
        )  # network architecture
    else:
        print("Unknown network architecture: ", network_arch)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.load_state_dict(torch.load(w))
    network.eval()  # set dropout and batch normalization layers to evaluation mode
    torch.no_grad()
    nr_parameters = network.count_parameters(trainable=True)

    # Load evaluation data
    test_images, test_labels_ds, bns, _, _ = u.loadData(
        data_dir + "/" + eval + "/",
        rf=rf,
        ss=ss,
        classes=classes,
        soft=False,
        uncertainty=False,
        crop_to_labels_of_interest=crop_image_to_non_zero_labels,
        pad_all=(network_arch != "dil2D"),
    )
    if "test" in eval:
        if "SegThor" in data_dir or "Brains" in data_dir:
            file_name = "segmentation"  # we have the ground truth segmentations
        else:
            file_name = ""  # we dont have the ground truth segmentations
    else:
        file_name = "segmentation"

    # obtain spacing of original images
    test_labels, spacings_orig, directions_orig = u.loadOriginalLabels(
        data_dir + "/" + eval + "/",
        bns,
        classes=classes,
        rf=rf,
        ss=ss,
        filename=file_name,
    )
    # spacing of analyzed images
    if "KiTS" in data_dir:
        spacing = (1.5, 1.5, 2.5)  # spacing of analyzed images
    elif "SegThor" in data_dir:
        spacing = (
            2.5,
            0.9765620231628418,
            0.9765620231628418,
        )  # spacing of analyzed images

    # make sure labels are valid when network is trained for background vs kidney+tumor
    if nclass == 2 and 3 in classes:  # binarize ground truth
        idx = np.where(classes == 3)
        classes[idx] = 1

    Dice = np.empty((len(test_images), nclass))  # array to store dices in per class
    HD = np.empty((len(test_images), nclass))  # array to store dices in per class
    HD_p = np.empty((len(test_images), nclass))  # array to store dices in per class
    ASD = np.empty((len(test_images), nclass))  # array to store dices in per class
    times = []
    for imidx in range(len(test_images)):
        bn = bns[imidx]
        print("Analyzing image: ", bn)
        im = test_images[imidx]
        im = np.asarray(im, dtype=np.float64)

        # prepare the data to work with. For 3D networks, we will process
        # the data a couple of times, each time moving the receptive field 1
        # voxel over and we then average the result

        # first make sure that we can fit multiples of the 3D receptive field:
        pad_z = patch_size_net - im.shape[0] % patch_size_net
        pad_y = patch_size_net - im.shape[1] % patch_size_net
        pad_x = patch_size_net - im.shape[2] % patch_size_net
        # now add a couple of voxels so that we can generate results with
        # the receptive field moved over:
        pad_z += number_of_avgs_for_prediction
        pad_y += number_of_avgs_for_prediction
        pad_x += number_of_avgs_for_prediction
        im = np.pad(im, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
        im = torch.from_numpy(im).type(torch.FloatTensor).cuda()

        if only_down_sampled_soft_probs:
            lab = test_labels_ds[imidx]
        else:
            lab = test_labels[imidx]
        spacing_orig = spacings_orig[imidx]
        direction_orig = directions_orig[imidx]

        cummulative_class_probabilities = np.zeros(
            (nclass, im.shape[0], im.shape[1], im.shape[2])
        )
        cummulative_class_logits = np.zeros(
            (nclass, im.shape[0], im.shape[1], im.shape[2])
        )
        cummulative_class_UP = np.zeros((nclass, im.shape[0], im.shape[1], im.shape[2]))

        print(
            "Calculating probabilities and logits for input image with offset: ", end=""
        )
        start = time.time()
        for i in range(number_of_avgs_for_prediction):
            (
                current_offset_class_prob,
                current_offset_class_logits,
                current_offset_class_UP,
            ) = get_network_output_full_image(
                im,
                patch_size_net,
                network,
                offset=i,
                nclass=nclass,
                network_arch=network_arch,
                UP=predict_uncertainty,
            )
            cummulative_class_probabilities += current_offset_class_prob
            cummulative_class_logits += current_offset_class_logits
            cummulative_class_UP += current_offset_class_UP
        print("")

        # now that we have gathered the predictions multiple times,
        # each time starting the predictions one voxel moved over, we
        # can average all the resulting probabilities and logits
        class_probs = cummulative_class_probabilities / number_of_avgs_for_prediction
        class_logits = cummulative_class_logits / number_of_avgs_for_prediction
        if predict_uncertainty:
            class_UP = cummulative_class_UP / number_of_avgs_for_prediction
            class_UP = class_UP[:, :-pad_z, :-pad_y, :-pad_x]
            out_im_UP = np.empty((nclass,) + lab.shape)

        # undo the padding step
        im = im[:-pad_z, :-pad_y, :-pad_x].cpu().detach().numpy()
        class_probs = class_probs[:, :-pad_z, :-pad_y, :-pad_x]
        class_logits = class_logits[:, :-pad_z, :-pad_y, :-pad_x]

        out_im_class = np.empty((nclass,) + lab.shape)

        if "Brains" in data_dir:
            spacing = spacings_orig[imidx]  # these images are not resampled

        print("Computing and saving final result...")
        all_output = []
        all_output_UP = []
        for n in range(nclass):
            out = np.squeeze(class_probs[n, :, :, :])
            if predict_uncertainty:
                out_UP = np.squeeze(class_UP[n, :, :, :])
            if not only_down_sampled_soft_probs:
                out = rn.resample_image(out, spacing, spacing_orig, lab.shape, order=1)
                if predict_uncertainty:
                    out_UP = rn.resample_image(
                        out_UP, spacing, spacing_orig, lab.shape, order=1
                    )
            all_output.append(out)  # Average over all slices for background
            if predict_uncertainty:
                out_im_UP[n] = out_UP
            out_im_class[n] = out
        if return_soft_probs:
            soft_probs.append(all_output)
        out_im_class = np.squeeze(np.argmax(out_im_class, axis=0))

        end = time.time()
        times.append(end - start)

        if not only_down_sampled_soft_probs:
            im_resampled = rn.resample_image(
                im, spacing, spacing_orig, lab.shape
            )  # resample back to original resolution

        if clip:  # mask output of network with ground truth segmentation
            lab_bin = lab > 0
            out_im_class = out_im_class * lab_bin

        if predict_uncertainty:
            for n in range(nclass):
                u.saveImage(
                    save_dir + "/" + bn + "_uncertainty_class" + str(n) + ".nii.gz",
                    out_im_UP[n],
                    spacing=spacing_orig,
                    direction=direction_orig,
                )

        im_name = (
            save_dir
            + "/"
            + bn
            + "_prediction_using_"
            + network_arch
            + "_"
            + loss_func
            + "_"
            + str(seed)
        )
        if not only_down_sampled_soft_probs:
            # save final result image
            u.saveImage(
                save_dir + "/" + bn + "_image.nii.gz",
                im_resampled,
                spacing=spacing_orig,
                direction=direction_orig,
            )
            u.saveImage(
                save_dir + "/" + bn + "_reference.nii.gz",
                lab,
                spacing=spacing_orig,
                direction=direction_orig,
            )
        if save_predictions and not only_down_sampled_soft_probs:
            u.saveImage(
                im_name + ".nii.gz",
                out_im_class,
                spacing=spacing_orig,
                direction=direction_orig,
            )
        elif save_predictions:
            u.saveImage(
                im_name + "_ds.nii.gz",
                out_im_class,
                spacing=spacing,
                direction=direction_orig,
            )

        all_filenames_soft_probs = []
        if return_soft_probs or return_filenames_of_softprobs:
            # if save_soft_probs and not return_only_ds_soft_probs:
            for n in range(nclass):
                soft_prob_filename = (
                    save_dir
                    + "/"
                    + bn
                    + "_soft_probs_"
                    + str(n)
                    + "_using_"
                    + network_arch
                    + "_"
                    + loss_func
                    + "_"
                    + str(seed)
                    + ".nii.gz"
                )
                if only_down_sampled_soft_probs:
                    soft_prob_filename = (
                        save_dir
                        + "/"
                        + bn
                        + "_soft_probs_"
                        + str(n)
                        + "_using_"
                        + network_arch
                        + "_"
                        + loss_func
                        + str(seed)
                        + "_ds.nii.gz"
                    )
                if save_soft_probs and not only_down_sampled_soft_probs:
                    u.saveImage(
                        soft_prob_filename,
                        all_output[n],
                        spacing=spacing_orig,
                        direction=direction_orig,
                    )
                elif save_soft_probs and only_down_sampled_soft_probs:
                    u.saveImage(
                        soft_prob_filename,
                        all_output[n],
                        spacing=spacing,
                        direction=direction_orig,
                    )
                all_filenames_soft_probs.append(soft_prob_filename)

        if return_filenames_of_softprobs:
            filenames_soft_probs.append(all_filenames_soft_probs)

        # compute_dice = "test" not in eval
        compute_dice = (
            "test" not in eval or "SegThor" in data_dir or "Brains" in data_dir
        )
        if "train" in eval:
            compute_dice = False
        if not only_down_sampled_soft_probs and compute_dice:
            i = 0
            for n in classes:
                av_surf_dist = 10000
                haus_d = 10000
                haus_dp = 10000
                pred = out_im_class == n
                # pred = retain_largest_components(pred, n = 1)
                ref = lab == n
                assert ref.shape == pred.shape
                dice = dice_score(ref, pred)
                Dice[imidx, i] = dice
                try:
                    av_surf_dist = average_surface_distance(ref, pred, spacing_orig)
                    haus_d = hausdorff_distance(ref, pred, spacing_orig)
                    haus_dp = hausdorff_distance(
                        ref, pred, spacing_orig, percentile=True
                    )
                except:
                    continue
                ASD[imidx, i] = av_surf_dist
                HD[imidx, i] = haus_d
                HD_p[imidx, i] = haus_dp
                i += 1
                print("Dice for class ", str(n), " is: ", str(dice))
                print(
                    "Average surface distance for class ",
                    str(n),
                    " is: ",
                    str(av_surf_dist),
                )
                print("Hausdorff Distance for class ", str(n), " is: ", str(haus_d))

    if not only_down_sampled_soft_probs:
        for n in range(nclass):
            print("Class: ", n)
            print("AVERAGE DICE: ", np.average(Dice[:, n]))
            print("STD DEV DICE: ", np.std(Dice[:, n]))
            print("MAX DICE: ", np.max(Dice[:, n]), np.argmax(Dice[:, n]))
            print("MIN DICE: ", np.min(Dice[:, n]), np.argmin(Dice[:, n]))
            print("")

            # print("Class: ", n)
            # print("AVERAGE ASD: ", np.average(ASD[:, n]))
            # print("STD DEV ASD: ", np.std(ASD[:, n]))
            # print("MAX ASD: ", np.max(ASD[:, n]), np.argmax(ASD[:, n]))
            # print("MIN ASD: ", np.min(ASD[:, n]), np.argmin(ASD[:, n]))
            # print("")
            #
            # print("Class: ", n)
            # print("AVERAGE HD: ", np.average(HD[:, n]))
            # print("STD DEV HD: ", np.std(HD[:, n]))
            # print("MAX HD: ", np.max(HD[:, n]), np.argmax(HD[:, n]))
            # print("MIN HD: ", np.min(HD[:, n]), np.argmin(HD[:, n]))
            # print("")

        df = pandas.DataFrame(Dice)
        df.to_excel(
            save_dir + "/DICE_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )
        df = pandas.DataFrame(bns)
        df.to_excel(save_dir + "/basenames.xlsx", index=False)
        df = pandas.DataFrame(ASD)
        df.to_excel(
            save_dir + "/ASD_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )

        df = pandas.DataFrame(HD)
        df.to_excel(
            save_dir + "/HD_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )
        df = pandas.DataFrame(HD_p)
        df.to_excel(
            save_dir + "/HD95_" + network_arch + "_" + loss_func + ".xlsx", index=False
        )
    df = pandas.DataFrame(times)
    df.to_excel(
        save_dir + "/InferenceTimes_" + network_arch + "_" + loss_func + ".xlsx",
        index=False,
    )

    max_gpu_usage = torch.cuda.max_memory_allocated(device=None)
    file_obj = open(save_dir + "/" + run_name + ".txt", "w")
    file_obj.write(run_name)
    file_obj.write("Max GPU usage: " + str(max_gpu_usage))
    file_obj.write("nr_parameters: " + str(nr_parameters))

    torch.cuda.reset_max_memory_allocated(device=None)

    if return_soft_probs:
        return soft_probs
    elif return_filenames_of_softprobs:
        return filenames_soft_probs
    else:
        return None


def get_network_output_full_image(
    im, patch_size_net, network, offset, nclass, network_arch, UP=False
):
    local_probabilites = np.zeros((nclass, im.shape[0], im.shape[1], im.shape[2]))
    local_logits = np.zeros((nclass, im.shape[0], im.shape[1], im.shape[2]))
    local_UP = np.zeros((nclass, im.shape[0], im.shape[1], im.shape[2]))

    print("%s.. " % str(offset), end="")

    # fill in the probabilities and logits by moving over the image:
    for z in range(im.shape[0] // patch_size_net):
        for y in range(im.shape[1] // patch_size_net):
            for x in range(im.shape[2] // patch_size_net):
                z_start = z * patch_size_net + offset
                z_stop = (z + 1) * patch_size_net + offset
                y_start = y * patch_size_net + offset
                y_stop = (y + 1) * patch_size_net + offset
                x_start = x * patch_size_net + offset
                x_stop = (x + 1) * patch_size_net + offset

                # extract the image block to be processed by the network:
                im_block = im[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                assert patch_size_net == im_block.shape[0]
                assert patch_size_net == im_block.shape[1]
                assert patch_size_net == im_block.shape[2]
                # reshape for processing:
                im_block = im_block.reshape(
                    (1, 1, patch_size_net, patch_size_net, patch_size_net)
                )
                if (
                    network_arch == "Unet3DSmallDS"
                    or network_arch == "Unet3DSmallOrigDS"
                ):
                    probs, logits, low_probs, mid_probs = network(im_block)
                elif UP:
                    probs, logits, UC_pred = network(im_block)
                else:
                    probs, logits = network(im_block)
                    UC_pred = probs
                probs = probs[0].cpu().detach().numpy()
                local_probabilites[
                    :, z_start:z_stop, y_start:y_stop, x_start:x_stop
                ] = probs
                logits = logits[0].cpu().detach().numpy()
                local_logits[:, z_start:z_stop, y_start:y_stop, x_start:x_stop] = logits
                UC_pred = UC_pred[0].cpu().detach().numpy()
                local_UP[:, z_start:z_stop, y_start:y_stop, x_start:x_stop] = UC_pred

    return local_probabilites, local_logits, local_UP


def inference(
    data_dir,
    weight_dir,
    save_dir,
    run,
    network_arch,
    eval,
    network_weights,
    receptive_field,
    sample_size,
    classes,
    clip,
    temperature,
    crop_image_to_non_zero_labels,
    number_of_3D_net_predictions,
    create_mask,
    predict_uncertainty,
):
    if "2D" in network_arch:
        Network2D(
            data_dir,
            save_dir,
            run,
            eval,
            network_weights,
            receptive_field,
            sample_size,
            classes,
            clip,
            temperature,
            crop_image_to_non_zero_labels,
            network_arch=network_arch,
            create_mask=create_mask,
            predict_uncertainty=predict_uncertainty,
        )  # 2D network architecture
    elif (
        network_arch == "Unet3D"
        or network_arch == "Unet3DStrided"
        or network_arch == "Resnet3D"
        or network_arch == "Resnet3DNEW"
        or network_arch == "Resnet3DUnc"
        or network_arch == "Unet3DSmall"
        or network_arch == "Unet3DSmallELU"
        or network_arch == "Unet3DSmallDS"
        or network_arch == "Unet3DSmallOrig"
        or network_arch == "Unet3DSmallOrigDS"
    ):
        Network3D(
            data_dir,
            save_dir,
            run,
            eval,
            network_arch,
            network_weights,
            receptive_field,
            sample_size,
            classes,
            clip,
            temperature,
            crop_image_to_non_zero_labels,
            number_of_3D_net_predictions,
            predict_uncertainty=predict_uncertainty,
        )
    else:
        print("Unknown network architecture, stop inference.")
        return


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    description = "Test a neural network for segmentation of the kidneys."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        type=str,
        help="Directory where data is stored",
    )
    parser.add_argument(
        "-w",
        "--weight_dir",
        required=False,
        type=str,
        help="Directory where weights are stored. Only"
        "used for the 3D network. For 2D networks,"
        "provide the full path to the network"
        "you want to use",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        required=True,
        type=str,
        help="Directory where results will be stored",
    )
    parser.add_argument(
        "-r", "--run", required=True, type=str, help="Name of the experiment"
    )
    parser.add_argument(
        "-n",
        "--network_arch",
        type=str,
        required=True,
        choices=[
            "dil2D",
            "Unet3D",
            "Unet3DStrided",
            "Resnet3D",
            "Resnet3DNEW",
            "Resnet3DUnc",
            "Unet3DSmall",
            "Unet3DSmallELU",
            "Unet3DSmallDS",
            "Unet3DSmallOrig",
            "Unet3DSmallOrigDS",
            "Resnet2DNEW",
            "Unet2DSmallOrig",
        ],
        default="dil2D",
        help="Network architecture: either 2D or 3D dilated network",
    )
    parser.add_argument(
        "-e",
        "--eval",
        type=str,
        required=True,
        choices=["val", "test", "train"],
        help="Evaluation on validation ('val'), test ('test') or training ('train') data.",
    )
    parser.add_argument(
        "--network-weights",
        dest="network_weights",
        type=str,
        required=True,
        help="Specify the path to the network weights used for testing",
    )
    parser.add_argument(
        "--receptive-field-size",
        type=int,
        required=False,
        default=131,
        dest="receptive_field",
        help="Network architecture: either use a 131x131 "
        "receptive field (largest dilation layer "
        "has a dilation of 32 voxels), or use a "
        "67x67 or 35x35 receptive field.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=False,
        default=54,
        dest="sample_size",
        help="Batch: specify the length "
        "of one of the sides of the (square) sample "
        "size area. Choices are 54 or 10 voxels",
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=False,
        default="0,1,2",
        help="specify the classes used during training in a "
        " comma separated string: 0=background, 1=kidney, "
        "2=tumor, 3=kidney+tumor",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        required=False,
        dest="temperature",
        default=1,
        help="Specify the temperature during calculation of the softmax output",
    )
    parser.add_argument(
        "--crop-images-to-non-zero-labels",
        required=False,
        default=False,
        dest="crop_images_to_non_zero_labels",
        action="store_true",
        help="Crop the images to only include the non-zero label values. Pad images accordingly.",
    )
    parser.add_argument(
        "--clip",
        type=bool,
        required=False,
        default=False,
        dest="clip",
        help="Specify whether you want the output of the network masked with ground truth segmentation",
    )
    parser.add_argument(
        "--number-of-3D-net-predictions",
        required=False,
        default=8,
        dest="number_of_3D_net_predictions",
        type=int,
        help="Specify how many times to run the 3D net over the image with each "
        "time a different offset to calculate predictions. All predictions"
        "are averaged in the end. This is to avoid sharp discontinuities "
        "on the borders of processed 3D patches.",
    )
    parser.add_argument(
        "--create_mask",
        type=bool,
        required=False,
        default=False,
        dest="create_mask",
        help="Specify whether you want masks generated from the segmentation",
    )
    parser.add_argument(
        "--predict_uncertainty",
        type=bool,
        required=False,
        default=False,
        dest="predict_uncertainty",
        help="Specify whether you want to predict uncertainty",
    )

    args = parser.parse_args()

    inference(
        data_dir=args.data_dir,
        weight_dir=args.weight_dir,
        save_dir=args.save_dir,
        run=args.run,
        network_arch=args.network_arch,
        eval=args.eval,
        network_weights=args.network_weights,
        receptive_field=args.receptive_field,
        sample_size=args.sample_size,
        classes=args.classes,
        clip=args.clip,
        temperature=args.temperature,
        crop_image_to_non_zero_labels=args.crop_images_to_non_zero_labels,
        number_of_3D_net_predictions=args.number_of_3D_net_predictions,
        create_mask=args.create_mask,
        predict_uncertainty=args.predict_uncertainty,
    )
