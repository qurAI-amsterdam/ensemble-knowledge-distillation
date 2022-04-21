import argparse
import os.path
import re
import sys
import time

import numpy as np
import pandas
import torch

import resampling_norounding as rn
import utils as u
from testing_network import (
    Network2D,
    Network3D,
    average_surface_distance,
    dice_score,
    hausdorff_distance,
)
from utils import loadImage, return_center_image


def zeroList(input_list):
    output_list = list(input_list)
    for x in range(len(input_list)):
        zero_im = np.zeros(input_list[x].shape)
        output_list[x] = zero_im

    return output_list


def networks_are_trained_on_same_classes(in_networks):
    """Returns whether the provided networks were trained in the same classes

    Assumption: the filename is constructed such that the classes used
    for training are stated before the balance ratio used. For instance:

    ...-cl-0-1-2-balance-ratio-...

    or

    ...-cl-0-3-balance-ratio...

    :param in_networks: list of paths to trained networks
    :return:
    """
    if len(in_networks) == 1:
        return True

    # use a regular expression to find the classes that are used
    # and that are delimited by "cl-" and "-balance"
    class_string = re.findall("cl-(.*?)-balance", in_networks[0])[0]

    for in_network in in_networks:
        curr_class_str = re.findall("cl-(.*?)-balance", in_network)[0]
        if not (curr_class_str == class_string):
            return False

    return True


def compute_mean_and_std_and_save_result(
    soft_probs, filenameprefix, filenamepostfix, spacing, direction
):
    np_soft_probs = np.asarray(soft_probs)
    np_soft_probs_std = np.std(np_soft_probs, axis=0)
    u.saveImage(
        filenameprefix + "std" + filenamepostfix,
        np_soft_probs_std,
        spacing=spacing,
        direction=direction,
    )
    np_soft_probs_mean = np.mean(np_soft_probs, axis=0)
    u.saveImage(
        filenameprefix + "mean" + filenamepostfix,
        np_soft_probs_mean,
        spacing=spacing,
        direction=direction,
    )
    return None


def ensemble(
    data_dir,
    save_dir,
    eval,
    clip,
    alpha,
    temperature,
    crop_image_to_non_zero_labels,
    number_of_3D_net_predictions,
    runs,
    only_produce_downsampled_soft_probs=False,
    be_memory_efficient=False,
):

    if not networks_are_trained_on_same_classes(runs):
        print(
            "The provided networks have not all been trained "
            "on the same classes. Provided networks:\n"
        )
        for in_network in runs:
            print(in_network)
        print("\nExiting...")
        sys.exit()

    classes = re.findall("cl-(.*?)-balance", runs[0])[0]
    classes_for_func_call = classes.replace("-", ",")
    classes = classes.split("-")
    classes = np.asarray(classes, dtype=int)
    nclass = len(classes)
    soft_outputs = []
    nr_runs = len(runs)

    # Obtain soft outputs for cases processed using all provided trained networks
    for i in range(nr_runs):
        run = runs[i]
        print("Run: ", run)
        experiment_name = os.path.basename(os.path.dirname(run))
        print(experiment_name)
        seed = experiment_name[:3]
        s = 2
        if not seed[0].isdigit():
            seed = 123
            s = 0
        network_arch = experiment_name.split("-")[s]
        sample_size = int(experiment_name.split("-")[s + 2])
        receptive_field = int(experiment_name.split("-")[s + 4])

        print("Processing Experiment: ", experiment_name)

        if "2D" in network_arch:
            soft_probs_per_network = Network2D(
                data_dir=data_dir,
                save_dir=save_dir,
                eval=eval,
                network_weights=run,
                receptive_field=receptive_field,
                sample_size=sample_size,
                classes=classes_for_func_call,
                clip=clip,
                temperature=temperature,
                crop_image_to_non_zero_labels=crop_image_to_non_zero_labels,
                return_soft_probs=(not be_memory_efficient),
                run=experiment_name,
                save_soft_probs=be_memory_efficient,
                save_predictions=False,
                return_only_ds_soft_probs=only_produce_downsampled_soft_probs,
                return_filenames_of_softprobs=be_memory_efficient,
                network_arch=network_arch,
                seed=seed,
            )
        else:
            soft_probs_per_network = Network3D(
                data_dir=data_dir,
                save_dir=save_dir,
                run=experiment_name,
                eval=eval,
                network_arch=network_arch,
                network_weights=run,
                receptive_field=receptive_field,
                sample_size=sample_size,
                classes=classes_for_func_call,
                clip=clip,
                temperature=temperature,
                crop_image_to_non_zero_labels=crop_image_to_non_zero_labels,
                return_soft_probs=(not be_memory_efficient),
                number_of_avgs_for_prediction=number_of_3D_net_predictions,
                save_soft_probs=be_memory_efficient,
                save_predictions=False,
                only_down_sampled_soft_probs=only_produce_downsampled_soft_probs,
                return_filenames_of_softprobs=be_memory_efficient,
                seed=seed,
            )

        soft_outputs.append(soft_probs_per_network)

    if be_memory_efficient:
        print("What we got back from the call to Network3D:")
        print(soft_outputs)

    # we only need the original labels now, so the receptive field and sample size here are unimportant.
    # we set them to some plausible defaults.
    test_images, test_labels_ds, bns, _, _ = u.loadData(
        data_dir + "/" + eval + "/",
        rf=0,
        ss=4,
        classes=classes,
        soft=False,
        uncertainty=False,
        crop_to_labels_of_interest=crop_image_to_non_zero_labels,
    )

    # the arguments receptive field and sample size are not at all used in the function
    # below, so their values do not matter.
    if "test" in eval:
        if "SegThor" in data_dir or "Brains" in data_dir:
            file_name = "segmentation"  # we have the ground truth segmentations
        else:
            file_name = ""  # we dont have the ground truth segmentations
    else:
        file_name = "segmentation"
    test_labels_orig, spacings_orig, directions_orig = u.loadOriginalLabels(
        data_dir + "/" + eval + "/",
        bns,
        classes=classes,
        rf=0,
        ss=4,
        filename=file_name,
    )

    # soft_outputs = np.asarray(soft_outputs)
    # Obtain Dice score for the ensemble
    Dice = np.empty((len(test_labels_orig), nclass))
    ASD = np.empty((len(test_labels_orig), nclass))
    HD = np.empty((len(test_labels_orig), nclass))
    HD95 = np.empty((len(test_labels_orig), nclass))
    times = []

    if nclass == 2 and 3 in classes:
        idx = np.where(classes == 3)
        classes[idx] = 1

    for imidx in range(len(test_labels_orig)):
        bn = bns[imidx]
        print("Image: ", bn)
        if only_produce_downsampled_soft_probs:
            lab = test_labels_ds[imidx]
            if "KiTS" in data_dir:
                spacing = (1.5, 1.5, 2.5)  # spacing of analyzed images
            elif "SegThor" in data_dir:
                spacing = (
                    2.5,
                    0.9765620231628418,
                    0.9765620231628418,
                )  # spacing of analyzed images
            else:
                spacing = spacings_orig[imidx]
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        else:
            lab = test_labels_orig[imidx]
            spacing = spacings_orig[imidx]
            direction = directions_orig[imidx]

        # compute average softmax output of the ensemble
        out_im_class = np.zeros((nclass,) + lab.shape)
        # out_im_class = np.zeros(((nclass,) + test_labels_orig[imidx].shape))
        perclass_soft_probs = list()

        start = time.time()
        for n in range(nr_runs):
            all_class_soft_probs = list()
            if not be_memory_efficient:
                for c in range(nclass):
                    soft_probs_this_run_c = soft_outputs[n][imidx][c]
                    if (
                        soft_probs_this_run_c.shape != lab.shape
                    ):  # check if input image was padded
                        soft_probs_this_run_c = return_center_image(
                            soft_probs_this_run_c, lab
                        )
                    # if only_produce_downsampled_soft_probs:
                    #     soft_probs_this_run_c = rn.resample_image(soft_probs_this_run_c, spacing, spacings_orig[imidx],
                    #                                           test_labels_orig[imidx].shape, order=1)
                    all_class_soft_probs.append(soft_probs_this_run_c)
            else:
                # read in the file from disk...
                for c in range(nclass):
                    soft_probs_this_run_c, _, _ = loadImage(soft_outputs[n][imidx][c])
                    if (
                        soft_probs_this_run_c.shape != lab.shape
                    ):  # check if input image was padded
                        soft_probs_this_run_c = return_center_image(
                            soft_probs_this_run_c, lab
                        )
                    # if only_produce_downsampled_soft_probs:
                    #     soft_probs_this_run_c = rn.resample_image(soft_probs_this_run_c, spacing, spacings_orig[imidx],
                    #                                           test_labels_orig[imidx].shape, order=1)
                    all_class_soft_probs.append(soft_probs_this_run_c)

            temp_soft_probs_perclass = list()
            for c in range(nclass):
                out_im_class[c] += all_class_soft_probs[c] / nr_runs
                temp_soft_probs_perclass.append(all_class_soft_probs[c])

            perclass_soft_probs.append(
                temp_soft_probs_perclass
            )  # this is a list of lists with the soft probabilities per class

        perclass_soft_probs = np.asarray(perclass_soft_probs)
        # calculate the variance in the soft predictions:
        for c in range(nclass):
            if only_produce_downsampled_soft_probs:
                filenamepostfix = "_" + str(classes[c]) + "_soft_probs_ds.nii.gz"
            else:
                filenamepostfix = "_" + str(classes[c]) + "_soft_probs.nii.gz"

            compute_mean_and_std_and_save_result(
                perclass_soft_probs[:, c],
                filenameprefix=save_dir + "/" + bn + "_ensemble_",
                filenamepostfix=filenamepostfix,
                spacing=spacing,
                direction=direction,
            )
            # compute_mean_and_std_and_save_result(perclass_soft_probs[:, c],
            #                             filenameprefix=save_dir + "/" + bn + "_ensemble_",
            #                             filenamepostfix=filenamepostfix,
            #                             spacing=spacings_orig[imidx], direction=directions_orig[imidx])

        if only_produce_downsampled_soft_probs:
            out_im_class_ds = np.copy(out_im_class)
            out_im_class = np.zeros((nclass,) + test_labels_orig[imidx].shape)
            for c in range(nclass):
                out_im_class[c] = rn.resample_image(
                    out_im_class_ds[c],
                    spacing,
                    spacings_orig[imidx],
                    test_labels_orig[imidx].shape,
                    order=1,
                )
        out_im_class = np.squeeze(np.argmax(out_im_class, axis=0))

        end = time.time()
        times.append(end - start)

        if clip:  # mask output of network with ground truth segmentation
            lab_bin = test_labels_orig[imidx] > 0
            # lab_bin = lab > 0
            out_im_class = out_im_class * lab_bin

        if only_produce_downsampled_soft_probs:
            # save_ext = "_ds"
            save_ext = "_fromds"
        else:
            save_ext = ""

        # u.saveImage(save_dir + "/" + bn + "_ensemble_prediction" + save_ext + ".nii.gz",
        #             out_im_class, spacing=spacing, direction=direction)
        # u.saveImage(save_dir + "/" + bn + "_ensemble_reference" + save_ext + ".nii.gz",
        #             lab, spacing=spacing, direction=direction)

        # if "test" in eval and "ACDC" in data_dir:
        #     out_im_class = out_im_class[5:-5, 5:-5, 5:-5]
        im_name = save_dir + "/" + bn + "_EnsemblePrediction"
        if "ACDC" in data_dir:
            im_name = save_dir + "/" + bn + "_ED"
        # u.saveImage(save_dir + "/" + bn + "_ensemble_prediction" + save_ext + ".nii.gz",
        u.saveImage(
            im_name + ".nii.gz",
            out_im_class,
            spacing=spacings_orig[imidx],
            direction=directions_orig[imidx],
        )
        u.saveImage(
            save_dir + "/" + bn + "_ensemble_reference" + save_ext + ".nii.gz",
            lab,
            spacing=spacings_orig[imidx],
            direction=directions_orig[imidx],
        )

        i = 0
        compute_dice = (
            "test" not in eval or "SegThor" in data_dir or "Brains" in data_dir
        )
        # compute_dice = "test" not in eval
        if "train" in eval:
            compute_dice = False
        if compute_dice:
            for n in classes:
                av_surf_dist = 10000
                haus_d = 10000
                haus_d95 = 10000
                pred = out_im_class == n
                # pred = retain_largest_components(pred, n = 1)
                # ref = lab == n
                ref = test_labels_orig[imidx] == n
                dice = dice_score(ref, pred)
                Dice[imidx, i] = dice
                try:
                    av_surf_dist = average_surface_distance(
                        ref, pred, spacings_orig[imidx]
                    )
                    haus_d = hausdorff_distance(ref, pred, spacings_orig[imidx])
                    haus_d95 = hausdorff_distance(
                        ref, pred, spacings_orig[imidx], percentile=True
                    )
                except:
                    continue

                ASD[imidx, i] = av_surf_dist
                HD[imidx, i] = haus_d
                HD95[imidx, i] = haus_d95
                i += 1
                print("Dice for class ", str(n), " is: ", str(dice))
                print(
                    "Average surface distance for class ",
                    str(n),
                    " is: ",
                    str(av_surf_dist),
                )
                print("Hausdorff Distance for class ", str(n), " is: ", str(haus_d))

    for n in range(nclass):
        print("Class: ", n)
        print("AVERAGE DICE: ", np.average(Dice[:, n]))
        print("STD DEV DICE: ", np.std(Dice[:, n]))
        print("MAX DICE: ", np.max(Dice[:, n]), np.argmax(Dice[:, n]))
        print("MIN DICE: ", np.min(Dice[:, n]), np.argmin(Dice[:, n]))
        print("")

        print("Class: ", n)
        print("AVERAGE ASD: ", np.average(ASD[:, n]))
        print("STD DEV ASD: ", np.std(ASD[:, n]))
        print("MAX ASD: ", np.max(ASD[:, n]), np.argmax(ASD[:, n]))
        print("MIN ASD: ", np.min(ASD[:, n]), np.argmin(ASD[:, n]))
        print("")

        print("Class: ", n)
        print("AVERAGE HD: ", np.average(HD[:, n]))
        print("STD DEV HD: ", np.std(HD[:, n]))
        print("MAX HD: ", np.max(HD[:, n]), np.argmax(HD[:, n]))
        print("MIN HD: ", np.min(HD[:, n]), np.argmin(HD[:, n]))
        print("")

    df = pandas.DataFrame(Dice)
    df.to_excel(save_dir + "/Ensemble_DICE.xlsx", index=False)
    df = pandas.DataFrame(bns)
    df.to_excel(save_dir + "/Ensemble_basenames.xlsx", index=False)

    df = pandas.DataFrame(ASD)
    df.to_excel(save_dir + "/Ensemble_ASD.xlsx", index=False)

    df = pandas.DataFrame(HD)
    df.to_excel(save_dir + "/Ensemble_HD.xlsx", index=False)

    df = pandas.DataFrame(HD95)
    df.to_excel(save_dir + "/Ensemble_HD95.xlsx", index=False)

    df = pandas.DataFrame(times)
    df.to_excel(save_dir + "/Ensemble_InferenceTimes.xlsx", index=False)


def get_list_of_networks_from_text(path_to_test):
    all_paths = []
    with open(path_to_test) as f:
        lines = f.read().splitlines()
        for line in lines:
            if os.path.exists(line):
                all_paths.append(line)

    return all_paths


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    description = (
        "Create an ensemble of networks while obtaining "
        "and saving the soft labels from the ensemble."
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--trained-networks",
        dest="trained_networks",
        type=str,
        required=True,
        help="Point to the text file that contains the paths "
        "to the trained networks. One network path per line.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        type=str,
        help="Directory where data is stored. This directory should "
        "contain the subdirectories train, val and test.",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        required=True,
        type=str,
        help="Parent directory where weights are stored. "
        "The weights will be stored in a subdirectory"
        "based on the chosen parameters.",
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
        "--clip",
        type=bool,
        required=False,
        default=False,
        dest="clip",
        help="Specify whether you want the output of the network masked with ground truth segmentation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        dest="alpha",
        default=0.1,
        help="Specify the weighting factor when using the soft labels. The function "
        "looks like: MSE_loss + alpha * CE_loss",
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
        "--only-generate-down-sampled-soft-probs",
        required=False,
        default=False,
        type=bool,
        dest="only_generate_down_sampled_soft_probs",
        help="Use this option if you are not interested in knowing what the "
        "performance of the ensemble is on the full sized data, but when "
        "instead you want to produce (save) down sampled versions of the "
        "soft probabilities. You can use those to train the distilled network.",
    )
    parser.add_argument(
        "--process-memory-efficient",
        required=False,
        default=False,
        type=bool,
        dest="process_memory_efficient",
        help="Don't keep all the calculated soft probabilities of the "
        "individual networks in memory. The amount of memory "
        "scales with the number of networks used and the number "
        "of files that are tested. This can get very excessive "
        "in terms of memory usage. Use this option to store all "
        "the soft probably on disk instead of using memory.",
    )
    args = parser.parse_args()

    trained_networks = get_list_of_networks_from_text(args.trained_networks)

    ensemble(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        eval=args.eval,
        clip=args.clip,
        alpha=args.alpha,
        temperature=args.temperature,
        crop_image_to_non_zero_labels=args.crop_images_to_non_zero_labels,
        number_of_3D_net_predictions=args.number_of_3D_net_predictions,
        runs=trained_networks,
        only_produce_downsampled_soft_probs=args.only_generate_down_sampled_soft_probs,
        be_memory_efficient=args.process_memory_efficient,
    )
