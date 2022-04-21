import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim

import networks as nw
import utils as u


def write_logfile(
    save_dir,
    run,
    random_seed,
    n,
    optimizer,
    criterion,
    iter,
    alpha,
    temperature,
    num_feat=32,
    rotation=False,
):
    """
    Write textfile containing informatiom on training
    :param save_dir: directory where file is saved
    :param run: name of the experiment
    :param rn: random seed
    :param n: network architecture
    :param optimizer: optimizer used during training
    :param criterion: loss-function
    :param iter: number of passed iterations
    :param alpha: alpha used for combined loss function
    :param temperature: temperature used during training in softmax layer
    :return:
    """
    log_path = save_dir + r"/callbacks/"
    try:  # check if directory exists, otherwise make it
        os.stat(log_path)
    except:
        os.mkdir(log_path)

    log_path = save_dir + r"/callbacks/" + run

    txtfile = open(log_path + ".txt", "w")
    txtfile.write("Experiment: " + run + "\n")
    txtfile.write("Random seed: " + str(random_seed) + "\n")
    txtfile.write("Network architecture: " + n + "\n")
    txtfile.write("Optimizer: " + str(optimizer) + "\n")
    txtfile.write("Loss: " + str(criterion) + "\n")
    txtfile.write("Number of iterations: " + str(iter) + "\n")
    txtfile.write("Alpha for combined loss function: " + str(alpha) + "\n")
    txtfile.write("Temperature used during training: " + str(temperature) + "\n")
    txtfile.write("Number of features used during training: " + str(num_feat) + "\n")
    txtfile.write(
        "Rotation as data augmentation during training 3D network: "
        + str(rotation)
        + "\n"
    )
    txtfile.close()


def train(
    data_dir,
    save_dir,
    network_arch,
    random_seed,
    total_num_iterations=250000,
    visdom_port=8008,
    receptive_field=131,
    sample_size=54,
    patch_inclusion_criteria="entire-sample-area",
    batch_size=40,
    percent_bg_full_bg=0,
    classes="0,1,2",
    balance_ratio="5,3,2",
    soft=False,
    alpha=0.1,
    temperature=1,
    crop_image_to_non_zero_labels=False,
    lossF="CE",
    num_feat=32,
    rotation=False,
    optimization="Adam",
    soft_loss="MSE",
    uncertainty_prediction=False,
    lossweights=None,
    initialiazation=False,
    init_weights=None,
    WD=False,
    ensemble_form="uni",
):
    """
    Training neural networks
    :param data_dir: Directory where data is stored
    :param save_dir: Directory where weights are stored
    :param run: Name of the experiment
    :param network_arch: Network architecture
    :param total_num_iterations: Total number of iterations to train the
        network for.
    :param visdom_port: Which visdom port number to use
    :return:
    """
    # set random seed

    if uncertainty_prediction:
        assert soft
        # Make sure you only predict uncertainties when you are also training with soft targets

    np.random.seed(random_seed)

    classes_string = classes.replace(",", "-")
    classes = classes.split(",")
    classes = np.asarray(classes, dtype=int)
    if lossweights is not None:
        lossweights_string = lossweights.replace(",", "-")
        lossweights = lossweights.split(",")
        lossweights = np.asarray(lossweights, dtype=int)
    else:
        lossweights = np.ones(len(classes))

    balance_ratio_string = balance_ratio.replace(",", "-")
    balance_ratio = balance_ratio.split(",")
    balance_ratio = np.asarray(balance_ratio, dtype=int)

    patch_size_net = receptive_field + sample_size

    nclass = len(classes)  # number of classes: 0 == Background, 1 == Kidney, 2 == Tumor
    rf = receptive_field  # receptive field
    ss = sample_size  # sample size --> ss==0 means center voxel is classified
    bs = batch_size  # batch size
    if network_arch == "dil2D":
        network = nw.DilatedNetwork2D(
            nclass,
            temperature=temperature,
            receptive_field=receptive_field,
            uncertainty_prediction=uncertainty_prediction,
        )  # network architecture
    elif network_arch == "Unet3D":
        if patch_size_net % 8 != 0:
            print("Patch size need to be a multiple of 8")
            return
        network = nw.UNet3D(
            nclass, temperature=temperature, num_feat=num_feat
        )  # network architecture
    elif network_arch == "Unet3DStrided":
        if patch_size_net % 8 != 0:
            print("Patch size need to be a multiple of 8")
            return
        network = nw.UNet3D_stridedConv(
            nclass, temperature=temperature
        )  # network architecture
    elif network_arch == "Unet3DSmall" or network_arch == "Unet3DSmallELU":
        if patch_size_net % 8 != 0:
            print("Patch size need to be a multiple of 8")
            return
        network = nw.UNet3DSmall(
            nclass, temperature=temperature, use_ELU=(network_arch == "Unet3DSmallELU")
        )
    elif network_arch == "Unet3DSmallDS":
        if patch_size_net % 8 != 0:
            print("Patch size need to be a multiple of 8")
            return
        network = nw.UNet3DSmallDS(nclass, temperature=temperature)
    elif (
        network_arch == "Unet3DSmallOrig"
        or network_arch == "Unet3DSmallOrigDS"
        or network_arch == "Unet2DSmallOrig"
    ):
        if patch_size_net % 8 != 0:
            print("Patch size need to be a multiple of 8")
            return
        if network_arch == "Unet3DSmallOrig" or network_arch == "Unet3DSmallOrigDS":
            network = nw.UNet3DSmallOrig(
                nclass,
                temperature=temperature,
                deep_supervision=(network_arch == "Unet3DSmallOrigDS"),
                uncertainty_prediction=uncertainty_prediction,
            )
        else:
            network = nw.UNet2DSmallOrig(
                nclass, temperature=temperature, deep_supervision=False
            )
    elif network_arch == "Unet3DSmallA":
        if rf != 88:
            print("The receptive field of this network needs to be 88.")
            return
        if ss != 4 and ((ss - 4) % 8 != 0):
            print(
                "The sample size for this network can only be 4 or 4 plus "
                "multiples of 8 (for instance 12, 20, 28 etc)"
            )
            return
        network = nw.UNet3DSmallA(nclass, temperature=temperature)
    elif (
        network_arch == "Resnet3D"
        or network_arch == "Resnet3DNEW"
        or network_arch == "Resnet2DNEW"
    ):
        if patch_size_net % 8 != 0:
            print("Patch size need to be a multiple of 8")
            return
        if network_arch == "Resnet3D" or network_arch == "Resnet3DNEW":
            network = nw.Resnet3D(
                nclass, temperature=temperature, architecture=network_arch
            )  # network architecture
        else:
            network = nw.Resnet2D(
                nclass, temperature=temperature, architecture=network_arch
            )  # network architecture
    elif network_arch == "Resnet3DUnc":
        if patch_size_net % 8 != 0:
            print("Patch size need to be a multiple of 8")
            return
        network = nw.Resnet3DUncertainty(
            nclass, temperature=temperature
        )  # network architecture

    if soft:
        if soft_loss == "MSE":
            soft_criterion = torch.nn.MSELoss()
        elif soft_loss == "CE":
            soft_criterion = u.cross_entropy_softtargets()
        if uncertainty_prediction:
            uncertainty_var_criterion = torch.nn.MSELoss()
            uncertainty_loss = "andUP_MSE"
        else:
            uncertainty_loss = ""
        criterion = torch.nn.CrossEntropyLoss()
        used_citerion = soft_loss + "andCE" + uncertainty_loss
    else:
        if lossF == "CE":
            lossweights = (
                torch.from_numpy(lossweights).type(torch.FloatTensor).to("cuda")
            )
            criterion = torch.nn.CrossEntropyLoss(weight=lossweights)
            used_citerion = "CE"
        elif lossF == "dice":
            criterion = u.dice_loss()
            used_citerion = "dice"
        else:
            print("Unknown loss function: ", lossF)

    run = (
        str(random_seed)
        + "-rs-"
        + network_arch
        + "-ss-"
        + str(ss)
        + "-rf-"
        + str(rf)
        + "-patch-inc-"
        + patch_inclusion_criteria
        + "-iter-"
        + str(total_num_iterations)
        + "-bs-"
        + str(bs)
        + "-cl-"
        + classes_string
        + "-balance-ratio-"
        + balance_ratio_string
        + "-T-"
        + str(temperature)
        + "-loss-"
        + used_citerion
        + "-crop-"
        + str(crop_image_to_non_zero_labels)
        + "-patch-size-"
        + str(patch_size_net)
        + "-optim-"
        + str(optimization)
    )
    if rotation:
        run += "-rot-" + str(rotation)
    if soft:
        run += "-alpha-" + str(alpha)
    if initialiazation:
        run += "-initialized-" + str(initialiazation)
    if WD:
        run += "-WD-TRUE"
    save_dir = os.path.join(save_dir, run)

    # create the output save directory if it does not exist:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(
            "Warning: you have already run a network with the same parameters. "
            "If you want to run it again, move or delete the previous results "
            "that are stored here:\n%s" % save_dir
        )
        sys.exit()

    with open(save_dir + r"/commandline_args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    if initialiazation:  # train with pretrained network
        assert init_weights != None
        network.load_state_dict(torch.load(init_weights))

    if optimization == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        if WD:
            optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-5)
    elif optimization == "SGD-scheduled":
        optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9)
        milestone1 = int((total_num_iterations / 100) * 3)
        milestone2 = int((total_num_iterations / 100) * 10)
        # milestone3 = int((total_num_iterations / 100) * 45)
        # milestone1 = 2000
        # milestone2 = 4000
        print(
            "Learning rate scheduler milestones at iterations: %s, %s"
            % (str(milestone1), str(milestone2))
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[milestone1, milestone2], gamma=0.1
        )
    else:
        print("Warning: unknown optimization method: %s" % optimization)
        sys.exit()

    iter = 0
    final_iter = total_num_iterations

    write_logfile(
        save_dir,
        run,
        random_seed,
        network_arch,
        optimizer,
        iter=iter,
        criterion=used_citerion,
        alpha=alpha,
        temperature=temperature,
        num_feat=num_feat,
        rotation=rotation,
    )

    torch.save(
        network.state_dict(), save_dir + r"/interrupted_model_initialized_only.pth"
    )

    # load data for training and validation
    images_train, labels_train, _, soft_probs_train, soft_vars_train = u.loadData(
        data_dir + "/train/",
        rf=rf,
        ss=ss,
        classes=classes,
        soft=soft,
        uncertainty=uncertainty_prediction,
        crop_to_labels_of_interest=crop_image_to_non_zero_labels,
        pad_all=(network_arch != "dil2D"),
        ensemble_form=ensemble_form,
    )

    images_val, labels_val, _, soft_probs_val, soft_vars_val = u.loadData(
        data_dir + "/val/",
        rf=rf,
        ss=ss,
        classes=classes,
        soft=soft,
        uncertainty=uncertainty_prediction,
        crop_to_labels_of_interest=crop_image_to_non_zero_labels,
        pad_all=(network_arch != "dil2D"),
        ensemble_form=ensemble_form,
    )

    # make sure labels are valid when network is trained for background vs foreground
    if nclass == 2 and 3 in classes:
        idx = np.where(classes == 3)
        classes[idx] = 1

    # create batch generator
    if "2D" in network_arch:
        batch_iter_training = u.balanced_batch_iterator2D(
            images_train,
            labels_train,
            soft_probs_train,
            soft_vars_train,
            soft,
            uncertainty_prediction,
            bs,
            ss,
            classes,
            rf,
            patch_inclusion_criteria=patch_inclusion_criteria,
            percent_bg_full_bg=percent_bg_full_bg,
            balance_ratio=balance_ratio,
            lossF=lossF,
            only_in_plane=("ACDC" in data_dir),
            network_arch=network_arch,
        )
        batch_iter_validation = u.balanced_batch_iterator2D(
            images_val,
            labels_val,
            soft_probs_val,
            soft_vars_val,
            soft,
            uncertainty_prediction,
            bs,
            ss,
            classes,
            rf,
            patch_inclusion_criteria=patch_inclusion_criteria,
            percent_bg_full_bg=percent_bg_full_bg,
            balance_ratio=balance_ratio,
            lossF=lossF,
            only_in_plane=("ACDC" in data_dir),
            network_arch=network_arch,
        )
    elif "3D" in network_arch:
        batch_iter_training = u.balanced_batch_iterator3D(
            images_train,
            labels_train,
            soft_probs_train,
            soft_vars_train,
            soft,
            uncertainty_prediction,
            bs,
            ss,
            classes,
            rf,
            patch_inclusion_criteria=patch_inclusion_criteria,
            percent_bg_full_bg=percent_bg_full_bg,
            balance_ratio=balance_ratio,
            lossF=lossF,
            rotation=rotation,
        )
        batch_iter_validation = u.balanced_batch_iterator3D(
            images_val,
            labels_val,
            soft_probs_val,
            soft_vars_val,
            soft,
            uncertainty_prediction,
            bs,
            ss,
            classes,
            rf,
            patch_inclusion_criteria=patch_inclusion_criteria,
            percent_bg_full_bg=percent_bg_full_bg,
            balance_ratio=balance_ratio,
            lossF=lossF,
            rotation=rotation,
        )

    # visdom
    # viz = visdom.Visdom(server=socket.gethostname(), env=run, port=visdom_port)
    # errwindow = viz.line(X=np.empty((1)), Y=np.empty((1, 2)), opts=dict(legend=['Train_dice', 'Val_dice']))
    # trainloss = np.empty(0 + int(final_iter))
    # trainloss[:0] = 0
    # valloss = np.empty(0 + int(final_iter))
    # valloss[:0] = 0

    # training
    try:
        for ims, labs, soft_probs, uncertainty_var in batch_iter_training:
            print("training iteration ", iter)
            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            if network_arch == "Unet3DSmallDS" or network_arch == "Unet3DSmallOrigDS":
                labs_segm, logits, low_segm, mid_segm = network(ims)
            elif network_arch == "Resnet3DUnc" or uncertainty_prediction:
                labs_segm, logits, uncertainty_pred = network(ims)
                uncertainty_loss = 0
                for n in range(nclass):
                    uncertainty_loss += uncertainty_var_criterion(
                        uncertainty_pred[:, n], uncertainty_var[:, n]
                    )
                # uncertainty_loss_bg = uncertainty_var_criterion(uncertainty_pred[:, 0], uncertainty_var[:, 0])
                # uncertainty_loss_kidney = uncertainty_var_criterion(uncertainty_pred[:, 1], uncertainty_var[:, 1])
                # uncertainty_loss_tumor = uncertainty_var_criterion(uncertainty_pred[:, 2], uncertainty_var[:, 2])
                # uncertainty_loss = uncertainty_loss_bg + uncertainty_loss_kidney + uncertainty_loss_tumor
            else:
                labs_segm, logits = network(ims)
                uncertainty_loss = 0
            if soft:
                # Input CE has size [N, C, ...] while targets have size [N, ...]
                # Therefore, for the soft labels we compute the CE for each class separately
                # and add the results for all classes
                soft_loss = 0
                for n in range(nclass):
                    soft_loss += soft_criterion(labs_segm[:, n], soft_probs[:, n])

                # soft_loss_bg = soft_criterion(labs_segm[:, 0], soft_probs[:, 0])
                # soft_loss_kidney = soft_criterion(labs_segm[:, 1], soft_probs[:, 1])
                # soft_loss_tumor = soft_criterion(labs_segm[:, 2], soft_probs[:, 2])
                # soft_loss = soft_loss_bg+soft_loss_kidney+soft_loss_tumor
                # MSE_loss = MSE_criterion(labs_segm, soft_probs)
                # Cross entropy loss needs as input parameters:
                # input: raw predictions for each of the classes
                # target: the target class for each of the input prediction
                hard_loss = criterion(labs_segm, labs)
                loss = (soft_loss + alpha * hard_loss) + uncertainty_loss

            else:
                if (
                    network_arch == "Unet3DSmallDS"
                    or network_arch == "Unet3DSmallOrigDS"
                ):
                    loss = (
                        criterion(labs_segm, labs)
                        + criterion(mid_segm, labs)
                        + criterion(low_segm, labs)
                    )
                else:
                    loss = criterion(labs_segm, labs)

            loss.backward()
            optimizer.step()

            if optimization == "SGD-scheduled":
                scheduler.step()

            if not (iter % 5000):  # Validation
                print("Loss: ", loss.item())
                ims_val, labs_val, soft_probs_val, uncertainty_var_val = next(
                    batch_iter_validation
                )
                if (
                    network_arch == "Unet3DSmallDS"
                    or network_arch == "Unet3DSmallOrigDS"
                ):
                    labs_segm_val, logits_val, low_segm_val, mid_segm_val = network(
                        ims_val
                    )
                elif network_arch == "Resnet3DUnc" or uncertainty_prediction:
                    labs_segm_val, logits_val, uncertainty_pred_val = network(ims_val)
                else:
                    labs_segm_val, logits_val = network(ims_val)

                if (
                    network_arch == "Unet3DSmallDS"
                    or network_arch == "Unet3DSmallOrigDS"
                ):
                    loss_val = (
                        criterion(labs_segm_val, labs_val)
                        + criterion(mid_segm_val, labs_val)
                        + criterion(low_segm_val, labs_val)
                    )
                else:
                    loss_val = criterion(labs_segm_val, labs_val)

                # trainloss[iter:iter + 50] = loss.cpu().detach().numpy()
                # valloss[iter:iter + 50] = loss_val.cpu().detach().numpy()
                # if("2D" in network_arch):
                #     valim = ims_val[0, 0]
                #     if lossF == 'CE':
                #         viz.contour(labs_val[0, :, :].cpu().detach().numpy(), win=2,
                #                     opts=dict(title="Validation reference"))
                #     else:
                #         viz.contour(np.argmax(np.squeeze(labs_val[0, :, :, :].cpu().detach().numpy()), axis=0), win=2,
                #                     opts=dict(title="Validation reference"))
                #     viz.contour(np.argmax(np.squeeze(labs_segm_val[0, :, :, :].cpu().detach().numpy()), axis=0),
                #                 win=3,
                #                 opts=dict(title="Validation automatic"))
                #     if rf > 0:
                #         valim = valim[rf // 2:(-rf // 2), rf // 2:(-rf // 2)]
                # else:
                #     valim = ims_val[0, 0, ims_val.shape[2] // 2]
                #     if lossF == 'CE':
                #         viz.contour(labs_val[0, labs_val.shape[2] // 2, :, :].cpu().detach().numpy(), win=2,opts=dict(title="Validation reference"))
                #     else:
                #         viz.contour(
                #             np.argmax(np.squeeze(labs_val[0, :, labs_val.shape[2] // 2, :, :].cpu().detach().numpy()),
                #                       axis=0), win=2, opts=dict(title="Validation reference"))
                #     viz.contour(np.argmax(
                #         np.squeeze(labs_segm_val[0, :, labs_segm_val.shape[2] // 2, :, :].cpu().detach().numpy()),
                #         axis=0), win=3, opts=dict(title="Validation automatic"))
                # viz.image((valim + 1024.0) / 4095.0 * 255, win=1, opts=dict(title="validation_input"))

                # plotlen = final_iter
                # vizX = np.zeros((min(iter+1, plotlen), 2))
                # vizX[:, 0] = range(max(iter+1 - plotlen, 0), iter+1)
                # vizX[:, 1] = range(max(iter+1 - plotlen, 0), iter+1)
                # vizX[np.isnan(vizX)] = 0.0
                # vizY = np.zeros((min(iter+1, plotlen), 2))
                # # print(trainloss)
                # # print(trainloss[max(iter - plotlen, 0):iter+1].flatten())
                # # print(valloss)
                # # print(valloss[max(iter - plotlen, 0):iter+1].flatten())
                # vizY[:, 0] = trainloss[max(iter+1 - plotlen, 0):iter+1].flatten()
                # vizY[:, 1] = valloss[max(iter+1 - plotlen, 0):iter+1].flatten()
                # vizY[np.isnan(vizY)] = 0.0
                # # print(vizX)
                # # print(vizY)
                # viz.line(win=errwindow, X=vizX, Y=vizY, update='replace')
                # # viz.updateTrace(win=errwindow, X=vizX, Y=vizY, append=False)

            if not (iter % 2500):
                torch.save(
                    network.state_dict(), save_dir + r"/Temp_" + str(iter) + ".pth"
                )

            iter += 1
            if iter >= final_iter:
                break

    except KeyboardInterrupt:
        print("interrupted")
        torch.save(
            network.state_dict(), save_dir + r"/interrupted_model_" + str(iter) + ".pth"
        )

    finally:
        torch.save(network.state_dict(), save_dir + r"/FINAL_" + str(iter) + ".pth")

    write_logfile(
        save_dir,
        run,
        random_seed,
        network_arch,
        optimizer,
        iter=iter,
        criterion=used_citerion,
        alpha=alpha,
        temperature=temperature,
        rotation=rotation,
    )


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    description = "Train a neural network for segmentation of the kidneys."
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
        "-s",
        "--save_dir",
        required=True,
        type=str,
        help="Parent directory where weights are stored. "
        "The weights will be stored in a subdirectory"
        "based on the chosen parameters.",
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
            "Unet3DSmall",
            "Unet3DSmallELU",
            "Unet3DSmallDS",
            "Unet3DSmallOrig",
            "Unet3DSmallOrigDS",
            "Resnet3D",
            "Resnet3DNEW",
            "Unet3DSmallA",
            "Resnet3DUnc",
            "Unet2DSmallOrig",
            "Resnet2DNEW",
        ],
        help="Network architecture: either 2D dilated network, 3D U-net or 3D Residual network",
    )
    parser.add_argument(
        "-x", "--random_seed", type=int, required=True, help="Setting random seed"
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100000,
        help="Total number of iterations to train the network for",
    )
    parser.add_argument(
        "--visdom-port",
        dest="visdom_port",
        type=int,
        default=8008,
        help="Specify the visdom port to use to follow training.",
    )
    parser.add_argument(
        "--receptive-field-size",
        type=int,
        required=False,
        default=131,
        dest="receptive_field",
        help="Network architecture dil2D: either use a 131x131 "
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
        "size area. Choices are 54 or 10 voxels for dil2D",
    )
    parser.add_argument(
        "--patch-inclusion-criteria",
        type=str,
        required=False,
        default="entire-sample-area",
        dest="patch_inclusion_criteria",
        choices=["entire-sample-area", "patch-center-voxel"],
        help="Batch: specify the patch inclusion criteria. The"
        "options are to include a patch if a voxel of the "
        "target class occurs in the entire sample area "
        "(entire-sample-area), or to only include a patch "
        "if the center voxel of the patch contains the "
        "target class (patch_center_voxel).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=40,
        dest="batch_size",
        help="Specify the number of patches to use per batch.",
    )
    parser.add_argument(
        "--percentage-bg-patch-full-bg",
        type=int,
        required=False,
        default=0,
        dest="percent_bg_full_bg",
        help="Specify what percentage of the background patches "
        "need to be 100 percent background. I.e., all voxels in "
        "the patch are background. Other background patches "
        "are forced to also contain other labels (to "
        "forcefully include boundaries).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=False,
        default="0,1,2",
        help="specify the classes used during training in a comma "
        "separated string: 0=background, 1=kidney, 2=tumor, 3=kidney+tumor",
    )
    parser.add_argument(
        "--balance-ratios",
        type=str,
        required=False,
        dest="balance_ratios",
        default="5,3,2",
        help="specify the ratios between classes used during training in a comma "
        "separated string: 5=50 percent background, 3=30 percent kidney, 2=20 percent tumor",
    )
    parser.add_argument(
        "--lossweights",
        type=str,
        required=False,
        dest="lossweights",
        default=None,
        help="specify the ratios between classes used during computation of the CE loss in a comma "
        "separated string",
    )
    parser.add_argument(
        "--soft",
        type=bool,
        required=False,
        dest="soft",
        default=False,
        help="specify whether you want to load also soft probabilities as labels",
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
        "-loss",
        "--loss",
        type=str,
        required=False,
        default="CE",
        choices=["dice", "CE"],
        help="Loss function for normal training: either Dice (dice) or Crossentropy (CE). "
        "Default is Crossentropy.",
    )
    parser.add_argument(
        "-softloss",
        "--softloss",
        type=str,
        required=False,
        default="MSE",
        choices=["MSE", "CE"],
        help="Loss function for training on soft-targets: either Mean Squared Error (MSE) or Crossentropy (CE). "
        "Default is Mean Squared Error.",
    )
    parser.add_argument(
        "-num_features",
        "--num_features",
        type=int,
        required=False,
        default=32,
        help="Number of features used per layer in Unet",
    )
    parser.add_argument(
        "-rot",
        "--rot",
        required=False,
        default=False,
        action="store_true",
        help="Rotation as data augmentation during training 3D network",
    )
    parser.add_argument(
        "-UP",
        "--UP",
        required=False,
        default=False,
        action="store_true",
        help="Also perform uncertainty prediction when doing prediction with soft targets",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default="Adam",
        dest="optimizer",
        choices=["Adam", "SGD-scheduled"],
        help="Specify which optimizer to use. If you choose for the SGD-scheduled optimizer, "
        "the learning rate will start at 0.1 for the first 3 percent of the total "
        "number of iterations, then drops to 0.01 until it reaches 30 percent of the "
        "iterations and drops to 0.001 at 45 percent of the total number of iterations.",
    )
    parser.add_argument(
        "-initialiazation",
        "--initialiazation",
        required=False,
        default=False,
        action="store_true",
        help="Indicates whether we have random initialization or we use a pretrained network",
    )
    parser.add_argument(
        "-init_weights",
        "--init_weights",
        type=str,
        required=False,
        help="Directory and file of weights used to initialize network",
    )
    parser.add_argument(
        "-WD",
        "--WD",
        required=False,
        default=False,
        action="store_true",
        help="Whether L2 regularization is used",
    )
    parser.add_argument(
        "--ensemble_form",
        type=str,
        choices=["uni", "div"],
        default="uni",
        help="Which soft labels need to be used: uniform ensemble or diverse ensemble.",
    )

    args = parser.parse_args()

    if args.percent_bg_full_bg < 0:
        args.percent_bg_full_bg = 0
    if args.percent_bg_full_bg > 100:
        args.percent_bg_full_bg = 100

    # set the PyTorch random seed
    torch.manual_seed(args.random_seed)

    train(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        network_arch=args.network_arch,
        random_seed=args.random_seed,
        total_num_iterations=args.iterations,
        visdom_port=args.visdom_port,
        receptive_field=args.receptive_field,
        sample_size=args.sample_size,
        patch_inclusion_criteria=args.patch_inclusion_criteria,
        batch_size=args.batch_size,
        percent_bg_full_bg=args.percent_bg_full_bg,
        classes=args.classes,
        balance_ratio=args.balance_ratios,
        soft=args.soft,
        alpha=args.alpha,
        temperature=args.temperature,
        crop_image_to_non_zero_labels=args.crop_images_to_non_zero_labels,
        lossF=args.loss,
        num_feat=args.num_features,
        rotation=args.rot,
        optimization=args.optimizer,
        soft_loss=args.softloss,
        uncertainty_prediction=args.UP,
        lossweights=args.lossweights,
        initialiazation=args.initialiazation,
        init_weights=args.init_weights,
        WD=args.WD,
        ensemble_form=args.ensemble_form,
    )
