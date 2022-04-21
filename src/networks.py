import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_temperature(y, temperature):
    output = F.softmax(y / temperature, dim=1)
    return output


def sigmoid(y):
    output = torch.sigmoid(y)
    return output


class DilatedNetwork2D(nn.Module):
    """
    2D fully convolutional network with Dilated convolutional layers
    Receptive field: 131 x 131
    """

    def __init__(
        self, nclass, temperature, receptive_field=131, uncertainty_prediction=False
    ):
        super().__init__()
        self.C = 32
        self.nclass = nclass

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature

        if not (
            receptive_field == 131 or receptive_field == 67 or receptive_field == 35
        ):
            raise ValueError(
                "Error: DilatedNetwork2D currently only handles "
                "receptive fields of 131x131, 67x67, or 35x35. Provided: %s"
                % str(receptive_field)
            )
        self.receptive_field = receptive_field
        self.uncertainty_prediction = uncertainty_prediction
        self.layers()

    def layers(self):
        self.inputl = nn.Sequential(nn.Conv2d(1, self.C, 3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.C, self.C, 3), nn.ReLU())

        self.dilconv1 = nn.Sequential(
            nn.Conv2d(self.C, self.C, 3, dilation=2), nn.ReLU()
        )
        self.dilconv2 = nn.Sequential(
            nn.Conv2d(self.C, self.C, 3, dilation=4), nn.ReLU()
        )
        self.dilconv3 = nn.Sequential(
            nn.Conv2d(self.C, self.C, 3, dilation=8), nn.ReLU()
        )
        if self.receptive_field >= 67:
            self.dilconv4 = nn.Sequential(
                nn.Conv2d(self.C, self.C, 3, dilation=16), nn.ReLU()
            )
            if self.receptive_field == 131:
                self.dilconv5 = nn.Sequential(
                    nn.Conv2d(self.C, self.C, 3, dilation=32), nn.ReLU()
                )

        self.conv3 = nn.Sequential(nn.Conv2d(self.C, self.C, 3), nn.ReLU())

        self.bn1 = nn.BatchNorm2d(num_features=self.C)
        self.do1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Sequential(nn.Conv2d(self.C, self.C * 3, 1), nn.ReLU())
        self.bn2 = nn.BatchNorm2d(num_features=self.C * 3)
        self.do2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Sequential(nn.Conv2d(self.C * 3, self.C * 6, 1), nn.ReLU())
        self.outputl = nn.Conv2d(self.C * 6, self.nclass, 1)

        if self.uncertainty_prediction:  # add layers for uncertainty prediction
            self.bn1_u = nn.BatchNorm2d(num_features=self.C)
            self.do1_u = nn.Dropout2d(p=0.5)
            self.fc1_u = nn.Sequential(nn.Conv2d(self.C, self.C * 3, 1), nn.ReLU())
            self.bn2_u = nn.BatchNorm2d(num_features=self.C * 3)
            self.do2_u = nn.Dropout2d(p=0.5)
            self.fc2_u = nn.Sequential(nn.Conv2d(self.C * 3, self.C * 6, 1), nn.ReLU())
            self.outputl_u = nn.Conv2d(self.C * 6, self.nclass, 1)

    def forward(self, x):
        x = self.inputl(x)
        x = self.conv2(x)
        x = self.dilconv1(x)
        x = self.dilconv2(x)
        x = self.dilconv3(x)
        if self.receptive_field >= 67:
            x = self.dilconv4(x)
            if self.receptive_field == 131:
                x = self.dilconv5(x)
        x = self.conv3(x)

        segm = self.bn1(x)
        segm = self.do1(segm)
        segm = self.fc1(segm)
        segm = self.bn2(segm)
        segm = self.do2(segm)
        segm = self.fc2(segm)
        segm_logits = self.outputl(segm)
        segm_softmax = softmax_temperature(segm_logits, self.temperature)

        if self.uncertainty_prediction:
            uncp = self.bn1_u(x)
            uncp = self.do1_u(uncp)
            uncp = self.fc1_u(uncp)
            uncp = self.bn2_u(uncp)
            uncp = self.do2_u(uncp)
            uncp = self.fc2_u(uncp)
            uncp_logits = self.outputl_u(uncp)
            uncp_softmax = softmax_temperature(uncp_logits, self.temperature)
            return segm_softmax, segm_logits, uncp_softmax
        return segm_softmax, segm_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )


class Conv3x3Small(nn.Module):
    def __init__(self, in_feat, out_feat, use_ELU=False):
        super().__init__()

        activation_layer = nn.ReLU()
        if use_ELU:
            activation_layer = nn.ELU()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
            activation_layer,
            nn.Dropout(p=0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=3, stride=1, padding=1),
            activation_layer,
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.deconv = nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        # print("forward in upsample - outputs after up: %s" % str(outputs.size()))
        outputs = self.deconv(inputs)
        # print("forward in upsample - outputs after deconv: %s" % str(outputs.size()))
        # print("forward in upsample - down_outputs - nothing done to this: %s" % str(down_outputs.size()))
        out = torch.cat([outputs, down_outputs], 1)
        return out


class UNet3DSmall(nn.Module):
    def __init__(self, nclass, temperature, use_ELU=False):
        super().__init__()
        self.num_feat = [32, 64, 128, 256]
        self.nclass = nclass
        self.use_ELU = use_ELU

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.layers()

    def layers(self):
        self.down1 = nn.Sequential(
            Conv3x3Small(1, self.num_feat[0], use_ELU=self.use_ELU)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(self.num_feat[0]),
            Conv3x3Small(self.num_feat[0], self.num_feat[1], use_ELU=self.use_ELU),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(self.num_feat[1]),
            Conv3x3Small(self.num_feat[1], self.num_feat[2], use_ELU=self.use_ELU),
        )

        self.bottom = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(self.num_feat[2]),
            Conv3x3Small(self.num_feat[2], self.num_feat[3], use_ELU=self.use_ELU),
            nn.BatchNorm3d(self.num_feat[3]),
        )

        self.up1 = UpSample(self.num_feat[3], self.num_feat[2])
        self.upconv1 = nn.Sequential(
            Conv3x3Small(
                self.num_feat[2] + self.num_feat[2],
                self.num_feat[2],
                use_ELU=self.use_ELU,
            ),
            nn.BatchNorm3d(self.num_feat[2]),
        )

        self.up2 = UpSample(self.num_feat[2], self.num_feat[1])
        self.upconv2 = nn.Sequential(
            Conv3x3Small(
                self.num_feat[1] + self.num_feat[1],
                self.num_feat[1],
                use_ELU=self.use_ELU,
            ),
            nn.BatchNorm3d(self.num_feat[1]),
        )

        self.up3 = UpSample(self.num_feat[1], self.num_feat[0])
        self.upconv3 = nn.Sequential(
            Conv3x3Small(
                self.num_feat[0] + self.num_feat[0],
                self.num_feat[0],
                use_ELU=self.use_ELU,
            ),
            nn.BatchNorm3d(self.num_feat[0]),
        )

        self.final = nn.Conv3d(self.num_feat[0], self.nclass, kernel_size=1)

    def forward(self, inputs):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print(down3_feat.size())
        bottom_feat = self.bottom(down3_feat)

        # print(bottom_feat.size())
        # print(down3_feat.size())
        up1_feat = self.up1(bottom_feat, down3_feat)
        # print(up1_feat.size())
        up1_feat = self.upconv1(up1_feat)
        # print(up1_feat.size())
        # print(down2_feat.size())
        up2_feat = self.up2(up1_feat, down2_feat)
        # print(up2_feat.size())
        up2_feat = self.upconv2(up2_feat)
        # print(up2_feat.size())
        # print(down1_feat.size())
        up3_feat = self.up3(up2_feat, down1_feat)
        # print(up3_feat.size())
        up3_feat = self.upconv3(up3_feat)
        # print(up3_feat.size())

        x_logits = self.final(up3_feat)
        x_softmax = softmax_temperature(x_logits, self.temperature)
        return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


################################################################################
# UNet3DSmallA
#
# Implementation (more similar to) https://arxiv.org/pdf/1606.06650.pdf
#
# Details: No padding used for the Conv3d blocks. So the output of the
#          network is smaller than the input. For instance:
#
#          Input 100x100x100 ---> output 12x12x12
#          Input 108x108x108 ---> output 20x20x20
#          Input 116x116x116 ---> output 28x28x28
#
#          Sequences: Conv3d -> BatchNorm3d -> ReLU

################################################################################


class Conv3dBatchNorm3dReLUTwice(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_feat, out_feat, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class TransposeConvWithCopyCrop(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.deconv = nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        # the input from the downwards stream in this setup is
        # larger than the input from the upsample stream. So we need
        # to copy and crop this information:
        down_outputs_clone = down_outputs.clone()
        diff_x = (down_outputs.size()[2] - outputs.size()[2]) / 2
        diff_y = (down_outputs.size()[3] - outputs.size()[3]) / 2
        diff_z = (down_outputs.size()[4] - outputs.size()[4]) / 2
        down_outputs_clone = down_outputs_clone[
            :,
            :,
            diff_x : down_outputs.size()[2] - diff_x,
            diff_y : down_outputs.size()[3] - diff_y,
            diff_z : down_outputs.size()[4] - diff_z,
        ]
        out = torch.cat([outputs, down_outputs_clone], 1)
        return out


class UNet3DSmallA(nn.Module):
    def __init__(self, nclass, temperature):
        super().__init__()
        self.num_feat = [32, 64, 128, 256]
        self.nclass = nclass

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.layers()

    def layers(self):
        self.down1 = Conv3dBatchNorm3dReLUTwice(1, self.num_feat[0])

        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Conv3dBatchNorm3dReLUTwice(self.num_feat[0], self.num_feat[1]),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Conv3dBatchNorm3dReLUTwice(self.num_feat[1], self.num_feat[2]),
        )

        self.bottom = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Conv3dBatchNorm3dReLUTwice(self.num_feat[2], self.num_feat[3]),
        )

        self.up1 = TransposeConvWithCopyCrop(self.num_feat[3], self.num_feat[2])
        self.upconv1 = Conv3dBatchNorm3dReLUTwice(
            self.num_feat[2] + self.num_feat[2], self.num_feat[2]
        )

        self.up2 = TransposeConvWithCopyCrop(self.num_feat[2], self.num_feat[1])
        self.upconv2 = Conv3dBatchNorm3dReLUTwice(
            self.num_feat[1] + self.num_feat[1], self.num_feat[1]
        )

        self.up3 = TransposeConvWithCopyCrop(self.num_feat[1], self.num_feat[0])
        self.upconv3 = Conv3dBatchNorm3dReLUTwice(
            self.num_feat[0] + self.num_feat[0], self.num_feat[0]
        )

        self.final = nn.Conv3d(self.num_feat[0], self.nclass, kernel_size=1)

    def forward(self, inputs):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)

        bottom_feat = self.bottom(down3_feat)

        up1_feat = self.up1(bottom_feat, down3_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down2_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down1_feat)
        up3_feat = self.upconv3(up3_feat)

        x_logits = self.final(up3_feat)
        x_softmax = softmax_temperature(x_logits, self.temperature)

        return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


################################################################################
# / UNet3DSmallA
################################################################################


################################################################################
# Implementation (more similar to) https://arxiv.org/pdf/1606.06650.pdf
################################################################################


class Conv3x3SmallOrig(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3SmallOrig_2D(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feat),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feat),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class TransposeConv(nn.Module):
    def __init__(self, in_feat, out_feat, two_d=False):
        super().__init__()

        if not two_d:
            self.deconv = nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.deconv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out


class ConvertToOutputOrig(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride, two_d=False):
        super().__init__()

        if not two_d:
            self.conv1 = nn.Conv3d(
                in_feat, out_feat, kernel_size=3, stride=1, padding=1
            )

            self.transconv = nn.ConvTranspose3d(
                out_feat, out_feat, kernel_size=kernel_size, stride=stride
            )
        else:
            self.conv1 = nn.Conv2d(
                in_feat, out_feat, kernel_size=3, stride=1, padding=1
            )

            self.transconv = nn.ConvTranspose2d(
                out_feat, out_feat, kernel_size=kernel_size, stride=stride
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.transconv(outputs)
        return outputs


class UNet3DSmallOrig(nn.Module):
    def __init__(
        self, nclass, temperature, deep_supervision=False, uncertainty_prediction=False
    ):
        super().__init__()
        self.num_feat = [32, 64, 128, 256]
        self.nclass = nclass
        self.deep_supervision = deep_supervision

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.uncertainty_prediction = uncertainty_prediction
        self.layers()

    def layers(self):
        self.down1 = nn.Sequential(Conv3x3SmallOrig(1, self.num_feat[0]))

        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Conv3x3SmallOrig(self.num_feat[0], self.num_feat[1]),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Conv3x3SmallOrig(self.num_feat[1], self.num_feat[2]),
        )

        self.bottom = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Conv3x3SmallOrig(self.num_feat[2], self.num_feat[3]),
        )

        self.up1 = TransposeConv(self.num_feat[3], self.num_feat[2])
        self.upconv1 = Conv3x3SmallOrig(
            self.num_feat[2] + self.num_feat[2], self.num_feat[2]
        )

        # potentially used when deep supervision is turned on:
        self.lowcis = ConvertToOutputOrig(self.num_feat[2], self.nclass, 4, 4)

        self.up2 = TransposeConv(self.num_feat[2], self.num_feat[1])
        self.upconv2 = Conv3x3SmallOrig(
            self.num_feat[1] + self.num_feat[1], self.num_feat[1]
        )

        # potentially used when deep supervision is turned on:
        self.midcis = ConvertToOutputOrig(self.num_feat[1], self.nclass, 2, 2)

        self.up3 = TransposeConv(self.num_feat[1], self.num_feat[0])
        self.upconv3 = Conv3x3SmallOrig(
            self.num_feat[0] + self.num_feat[0], self.num_feat[0]
        )

        self.final = nn.Conv3d(self.num_feat[0], self.nclass, kernel_size=1)

        if self.uncertainty_prediction:
            self.final_u = nn.Conv3d(self.num_feat[0], self.nclass, kernel_size=1)

    def forward(self, inputs):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print(down3_feat.size())
        bottom_feat = self.bottom(down3_feat)

        # print(bottom_feat.size())
        # print(down3_feat.size())
        up1_feat = self.up1(bottom_feat, down3_feat)
        # print(up1_feat.size())
        up1_feat = self.upconv1(up1_feat)

        if self.deep_supervision:
            low_cis_logits = self.lowcis(up1_feat)

        # print(up1_feat.size())
        # print(down2_feat.size())
        up2_feat = self.up2(up1_feat, down2_feat)
        # print(up2_feat.size())
        up2_feat = self.upconv2(up2_feat)

        if self.deep_supervision:
            mid_cis_logits = self.midcis(up2_feat)

        # print(up2_feat.size())
        # print(down1_feat.size())
        up3_feat = self.up3(up2_feat, down1_feat)
        # print(up3_feat.size())
        up3_feat = self.upconv3(up3_feat)
        # print(up3_feat.size())

        x_logits = self.final(up3_feat)
        x_softmax = softmax_temperature(x_logits, self.temperature)
        if self.uncertainty_prediction:
            x_logits_u = self.final_u(up3_feat)
            x_softmax_u = softmax_temperature(x_logits_u, self.temperature)
            return x_softmax, x_logits, x_softmax_u

        if self.deep_supervision:
            low_cis_softmax = softmax_temperature(low_cis_logits, self.temperature)
            mid_cis_softmax = softmax_temperature(mid_cis_logits, self.temperature)
            return x_softmax, x_logits, low_cis_softmax, mid_cis_softmax
        else:
            return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


################################################################################
# / Implementation (more similar to) https://arxiv.org/pdf/1606.06650.pdf
################################################################################


class UNet2DSmallOrig(nn.Module):
    def __init__(self, nclass, temperature, deep_supervision=False):
        super().__init__()
        self.num_feat = [32, 64, 128, 256]
        self.nclass = nclass
        self.deep_supervision = deep_supervision

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.layers()

    def layers(self):
        self.down1 = nn.Sequential(Conv3x3SmallOrig_2D(1, self.num_feat[0]))

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Conv3x3SmallOrig_2D(self.num_feat[0], self.num_feat[1]),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Conv3x3SmallOrig_2D(self.num_feat[1], self.num_feat[2]),
        )

        self.bottom = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Conv3x3SmallOrig_2D(self.num_feat[2], self.num_feat[3]),
        )

        self.up1 = TransposeConv(self.num_feat[3], self.num_feat[2], two_d=True)
        self.upconv1 = Conv3x3SmallOrig_2D(
            self.num_feat[2] + self.num_feat[2], self.num_feat[2]
        )

        # potentially used when deep supervision is turned on:
        self.lowcis = ConvertToOutputOrig(
            self.num_feat[2], self.nclass, 4, 4, two_d=True
        )

        self.up2 = TransposeConv(self.num_feat[2], self.num_feat[1], two_d=True)
        self.upconv2 = Conv3x3SmallOrig_2D(
            self.num_feat[1] + self.num_feat[1], self.num_feat[1]
        )

        # potentially used when deep supervision is turned on:
        self.midcis = ConvertToOutputOrig(
            self.num_feat[1], self.nclass, 2, 2, two_d=True
        )

        self.up3 = TransposeConv(self.num_feat[1], self.num_feat[0], two_d=True)
        self.upconv3 = Conv3x3SmallOrig_2D(
            self.num_feat[0] + self.num_feat[0], self.num_feat[0]
        )

        self.final = nn.Conv2d(self.num_feat[0], self.nclass, kernel_size=1)

    def forward(self, inputs):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print(down3_feat.size())
        bottom_feat = self.bottom(down3_feat)

        # print(bottom_feat.size())
        # print(down3_feat.size())
        up1_feat = self.up1(bottom_feat, down3_feat)
        # print(up1_feat.size())
        up1_feat = self.upconv1(up1_feat)

        if self.deep_supervision:
            low_cis_logits = self.lowcis(up1_feat)

        # print(up1_feat.size())
        # print(down2_feat.size())
        up2_feat = self.up2(up1_feat, down2_feat)
        # print(up2_feat.size())
        up2_feat = self.upconv2(up2_feat)

        if self.deep_supervision:
            mid_cis_logits = self.midcis(up2_feat)

        # print(up2_feat.size())
        # print(down1_feat.size())
        up3_feat = self.up3(up2_feat, down1_feat)
        # print(up3_feat.size())
        up3_feat = self.upconv3(up3_feat)
        # print(up3_feat.size())

        x_logits = self.final(up3_feat)
        x_softmax = softmax_temperature(x_logits, self.temperature)

        if self.deep_supervision:
            low_cis_softmax = softmax_temperature(low_cis_logits, self.temperature)
            mid_cis_softmax = softmax_temperature(mid_cis_logits, self.temperature)
            return x_softmax, x_logits, low_cis_softmax, mid_cis_softmax
        else:
            return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


class ConvertToOutput(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        self.transconv = nn.ConvTranspose3d(
            out_feat, out_feat, kernel_size=kernel_size, stride=stride
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.transconv(outputs)
        return outputs


class UNet3DSmallDS(nn.Module):
    """UNet3DSmallDS - Deep Supervision

    Has additional outputs in the mid layer and lower layer
    """

    def __init__(self, nclass, temperature, use_ELU=False):
        super().__init__()
        self.num_feat = [32, 64, 128, 256]
        self.nclass = nclass
        self.use_ELU = use_ELU

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.layers()

    def layers(self):
        self.down1 = nn.Sequential(
            Conv3x3Small(1, self.num_feat[0], use_ELU=self.use_ELU)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(self.num_feat[0]),
            Conv3x3Small(self.num_feat[0], self.num_feat[1], use_ELU=self.use_ELU),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(self.num_feat[1]),
            Conv3x3Small(self.num_feat[1], self.num_feat[2], use_ELU=self.use_ELU),
        )

        self.bottom = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(self.num_feat[2]),
            Conv3x3Small(self.num_feat[2], self.num_feat[3], use_ELU=self.use_ELU),
            nn.BatchNorm3d(self.num_feat[3]),
        )

        self.up1 = UpSample(self.num_feat[3], self.num_feat[2])
        self.upconv1 = nn.Sequential(
            Conv3x3Small(
                self.num_feat[2] + self.num_feat[2],
                self.num_feat[2],
                use_ELU=self.use_ELU,
            ),
            nn.BatchNorm3d(self.num_feat[2]),
        )

        self.lowcis = ConvertToOutput(self.num_feat[2], self.nclass, 4, 4)

        self.up2 = UpSample(self.num_feat[2], self.num_feat[1])
        self.upconv2 = nn.Sequential(
            Conv3x3Small(
                self.num_feat[1] + self.num_feat[1],
                self.num_feat[1],
                use_ELU=self.use_ELU,
            ),
            nn.BatchNorm3d(self.num_feat[1]),
        )

        self.midcis = ConvertToOutput(self.num_feat[1], self.nclass, 2, 2)

        self.up3 = UpSample(self.num_feat[1], self.num_feat[0])
        self.upconv3 = nn.Sequential(
            Conv3x3Small(
                self.num_feat[0] + self.num_feat[0],
                self.num_feat[0],
                use_ELU=self.use_ELU,
            ),
            nn.BatchNorm3d(self.num_feat[0]),
        )

        self.final = nn.Conv3d(self.num_feat[0], self.nclass, kernel_size=1)

    def forward(self, inputs):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print(down3_feat.size())
        bottom_feat = self.bottom(down3_feat)

        # print(bottom_feat.size())
        # print(down3_feat.size())
        up1_feat = self.up1(bottom_feat, down3_feat)
        # print(up1_feat.size())
        up1_feat = self.upconv1(up1_feat)

        # convolve and transpose convolve the ouput in the lower
        # layer to produce an output segmentation there
        low_cis_logits = self.lowcis(up1_feat)

        # print(up1_feat.size())
        # print(down2_feat.size())
        up2_feat = self.up2(up1_feat, down2_feat)
        # print(up2_feat.size())
        up2_feat = self.upconv2(up2_feat)

        # convolve and transpose convolve the ouput in the middle
        # layer to produce an output segmentation there
        mid_cis_logits = self.midcis(up2_feat)

        # print(up2_feat.size())
        # print(down1_feat.size())
        up3_feat = self.up3(up2_feat, down1_feat)
        # print(up3_feat.size())
        up3_feat = self.upconv3(up3_feat)
        # print(up3_feat.size())

        x_logits = self.final(up3_feat)
        x_softmax = softmax_temperature(x_logits, self.temperature)

        low_cis_softmax = softmax_temperature(low_cis_logits, self.temperature)
        mid_cis_softmax = softmax_temperature(mid_cis_logits, self.temperature)

        return x_softmax, x_logits, low_cis_softmax, mid_cis_softmax

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


class UNet3D(nn.Module):
    def __init__(self, nclass, temperature, num_feat=32):
        super().__init__()
        # receptive field of 68
        self.C = num_feat
        self.nclass = nclass

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.layers()

    def layers(self):
        self.dconv_down1 = double_conv(1, self.C // 2)
        self.dconv_down2 = double_conv(self.C // 2, self.C)
        self.dconv_down3 = double_conv(self.C, self.C * 2)
        self.dconv_down4 = double_conv(self.C * 2, self.C * 2)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )

        self.dconv_up3 = double_conv(self.C * 4, self.C * 2)
        self.dconv_up2 = double_conv(self.C * 3, self.C)
        self.dconv_up1 = double_conv(int(self.C * 1.5), self.C // 2)

        self.bn1 = nn.BatchNorm3d(num_features=self.C // 2)
        self.do1 = nn.Dropout3d(p=0.5)
        self.fconv1 = nn.Sequential(
            nn.Conv3d(self.C // 2, self.C // 2, 1), nn.ReLU()
        )  # added non-linearity
        self.bn2 = nn.BatchNorm3d(num_features=self.C // 2)
        self.do2 = nn.Dropout3d(p=0.5)
        self.fconv2 = nn.Sequential(
            nn.Conv3d(self.C // 2, self.C, 1), nn.ReLU()
        )  # added fully connected layer
        self.conv_last = nn.Conv3d(self.C, self.nclass, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.fconv1(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.fconv2(x)
        x_logits = self.conv_last(x)
        x_softmax = softmax_temperature(x_logits, self.temperature)
        return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


class UNet3D_stridedConv(nn.Module):
    def __init__(self, nclass, temperature):
        super().__init__()
        # receptive field of 68
        self.C = 32
        self.nclass = nclass

        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.layers()

    def layers(self):
        self.dconv_down1 = double_conv(1, self.C // 2)
        self.dconv_down2 = double_conv(self.C // 2, self.C)
        self.dconv_down3 = double_conv(self.C, self.C * 2)
        self.dconv_down4 = double_conv(self.C * 2, self.C * 2)

        self.maxpool1 = nn.Sequential(
            nn.Conv3d(self.C // 2, self.C // 2, kernel_size=2, stride=2), nn.ReLU()
        )
        self.maxpool2 = nn.Sequential(
            nn.Conv3d(self.C, self.C, kernel_size=2, stride=2), nn.ReLU()
        )
        self.maxpool3 = nn.Sequential(
            nn.Conv3d(self.C * 2, self.C * 2, kernel_size=2, stride=2), nn.ReLU()
        )

        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )

        self.dconv_up3 = double_conv(self.C * 4, self.C * 2)
        self.dconv_up2 = double_conv(self.C * 3, self.C)
        self.dconv_up1 = double_conv(int(self.C * 1.5), self.C // 2)

        self.bn1 = nn.BatchNorm3d(num_features=self.C // 2)
        self.do1 = nn.Dropout3d(p=0.5)
        self.fconv1 = nn.Sequential(nn.Conv3d(self.C // 2, self.C // 2, 1), nn.ReLU())
        self.bn2 = nn.BatchNorm3d(num_features=self.C // 2)
        self.do2 = nn.Dropout3d(p=0.5)
        self.fconv2 = nn.Sequential(nn.Conv3d(self.C // 2, self.C, 1), nn.ReLU())
        self.conv_last = nn.Conv3d(self.C, self.nclass, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.fconv1(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.fconv2(x)
        x_logits = self.conv_last(x)
        x_softmax = softmax_temperature(x_logits, self.temperature)
        return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


class ResnetBlock3D(nn.Module):
    """
    Modified resnetblock as in Identity Mappings in Deep Residual Networks (He et al. 2016): fully pre-activated
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert padding_type == "zero"
        p = 1

        conv_block += [
            norm_layer(dim, affine=True),
            nn.ReLU(True),
            nn.Conv3d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim, affine=True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(True), nn.Conv3d(dim, dim, kernel_size=3, padding=p)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetBlock2D(nn.Module):
    """
    Modified resnetblock as in Identity Mappings in Deep Residual Networks (He et al. 2016): fully pre-activated
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert padding_type == "zero"
        p = 1

        conv_block += [
            norm_layer(dim, affine=True),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim, affine=True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(True), nn.Conv2d(dim, dim, kernel_size=3, padding=p)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Resnet3D(nn.Module):
    def __init__(self, nclass, temperature, architecture):
        super().__init__()
        self.C = 32
        self.norm_layer = nn.BatchNorm3d
        self.n_blocks = 6  # moet groter dan 0 zijn
        self.n_downsampling = 2
        self.first_kern = 3
        self.use_dropout = True
        self.nclass = nclass
        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        if architecture == "Resnet3D":
            self.layers()
        elif architecture == "Resnet3DNEW":
            self.layersnew()
        else:
            raise ValueError(
                "Error: undefined architecture. Provided: %s" % str(architecture)
            )

    def layers(self):
        model = [
            nn.Conv3d(1, self.C, kernel_size=self.first_kern, padding=1),
            self.norm_layer(self.C, affine=True),
            nn.ReLU(True),
        ]

        for i in range(self.n_downsampling):
            mult = 2**i
            model += [
                nn.Conv3d(
                    self.C * mult, self.C * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                self.norm_layer(self.C * mult * 2, affine=True),
                nn.ReLU(True),
            ]

        mult = 2**self.n_downsampling
        for i in range(self.n_blocks):
            model += [
                ResnetBlock3D(
                    self.C * mult,
                    "zero",
                    norm_layer=self.norm_layer,
                    use_dropout=self.use_dropout,
                )
            ]

        for i in range(self.n_downsampling):
            mult = 2 ** (self.n_downsampling - i)
            model += [
                nn.ConvTranspose3d(
                    self.C * mult,
                    int(self.C * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                self.norm_layer(int(self.C * mult / 2), affine=True),
                nn.ReLU(True),
            ]

        # model += [nn.Conv3d(self.C, cfg.CNUM, kernel_size=7, padding=3)]
        model += [nn.Dropout3d(p=0.5)]
        model += [nn.Conv3d(self.C, self.C, kernel_size=1)]
        model += [nn.Dropout3d(p=0.5)]
        model += [nn.Conv3d(self.C, self.nclass, kernel_size=1)]
        # model += [nn.Softmax()]

        self.model = nn.Sequential(*model)

    def layersnew(self):
        model = [
            nn.Conv3d(1, self.C, kernel_size=self.first_kern, padding=1),
            self.norm_layer(self.C, affine=True),
            nn.ReLU(True),
        ]

        for i in range(self.n_downsampling):
            mult = 2**i
            model += [
                nn.Conv3d(
                    self.C * mult, self.C * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                self.norm_layer(self.C * mult * 2, affine=True),
                nn.ReLU(True),
            ]

        mult = 2**self.n_downsampling
        for i in range(self.n_blocks):
            model += [
                ResnetBlock3D(
                    self.C * mult,
                    "zero",
                    norm_layer=self.norm_layer,
                    use_dropout=self.use_dropout,
                )
            ]

        for i in range(self.n_downsampling):
            mult = 2 ** (self.n_downsampling - i)
            model += [
                nn.ConvTranspose3d(
                    self.C * mult,
                    int(self.C * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                self.norm_layer(int(self.C * mult / 2), affine=True),
                nn.ReLU(True),
            ]

        # model += [nn.Conv3d(self.C, cfg.CNUM, kernel_size=7, padding=3)]
        model += [nn.Dropout3d(p=0.5)]
        model += [
            nn.Conv3d(self.C, self.C, kernel_size=1),
            nn.ReLU(True),
        ]  # added this non-linearity
        model += [nn.Dropout3d(p=0.5)]
        model += [nn.Conv3d(self.C, self.nclass, kernel_size=1)]
        # model += [nn.Softmax()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x_logits = self.model(input)
        x_softmax = softmax_temperature(x_logits, self.temperature)
        return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


class Resnet2D(nn.Module):
    def __init__(self, nclass, temperature, architecture):
        super().__init__()
        self.C = 32
        self.norm_layer = nn.BatchNorm2d
        self.n_blocks = 6  # moet groter dan 0 zijn
        self.n_downsampling = 2
        self.first_kern = 3
        self.use_dropout = True
        self.nclass = nclass
        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        if architecture == "Resnet2D":
            self.layers()
        elif architecture == "Resnet2DNEW":
            self.layersnew()
        else:
            raise ValueError(
                "Error: undefined architecture. Provided: %s" % str(architecture)
            )

    def layers(self):
        model = [
            nn.Conv2d(1, self.C, kernel_size=self.first_kern, padding=1),
            self.norm_layer(self.C, affine=True),
            nn.ReLU(True),
        ]

        for i in range(self.n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    self.C * mult, self.C * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                self.norm_layer(self.C * mult * 2, affine=True),
                nn.ReLU(True),
            ]

        mult = 2**self.n_downsampling
        for i in range(self.n_blocks):
            model += [
                ResnetBlock2D(
                    self.C * mult,
                    "zero",
                    norm_layer=self.norm_layer,
                    use_dropout=self.use_dropout,
                )
            ]

        for i in range(self.n_downsampling):
            mult = 2 ** (self.n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    self.C * mult,
                    int(self.C * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                self.norm_layer(int(self.C * mult / 2), affine=True),
                nn.ReLU(True),
            ]

        model += [nn.Dropout2d(p=0.5)]
        model += [nn.Conv2d(self.C, self.C, kernel_size=1)]
        model += [nn.Dropout2d(p=0.5)]
        model += [nn.Conv2d(self.C, self.nclass, kernel_size=1)]

        self.model = nn.Sequential(*model)

    def layersnew(self):
        model = [
            nn.Conv2d(1, self.C, kernel_size=self.first_kern, padding=1),
            self.norm_layer(self.C, affine=True),
            nn.ReLU(True),
        ]

        for i in range(self.n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    self.C * mult, self.C * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                self.norm_layer(self.C * mult * 2, affine=True),
                nn.ReLU(True),
            ]

        mult = 2**self.n_downsampling
        for i in range(self.n_blocks):
            model += [
                ResnetBlock2D(
                    self.C * mult,
                    "zero",
                    norm_layer=self.norm_layer,
                    use_dropout=self.use_dropout,
                )
            ]

        for i in range(self.n_downsampling):
            mult = 2 ** (self.n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    self.C * mult,
                    int(self.C * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                self.norm_layer(int(self.C * mult / 2), affine=True),
                nn.ReLU(True),
            ]

        model += [nn.Dropout2d(p=0.5)]
        model += [
            nn.Conv2d(self.C, self.C, kernel_size=1),
            nn.ReLU(True),
        ]  # added this non-linearity
        model += [nn.Dropout2d(p=0.5)]
        model += [nn.Conv2d(self.C, self.nclass, kernel_size=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x_logits = self.model(input)
        x_softmax = softmax_temperature(x_logits, self.temperature)
        return x_softmax, x_logits

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


class Resnet3DUncertainty(nn.Module):
    def __init__(self, nclass, temperature):
        super().__init__()
        self.C = 32
        self.norm_layer = nn.BatchNorm3d
        self.n_blocks = 6  # moet groter dan 0 zijn
        self.n_downsampling = 2
        self.first_kern = 3
        self.use_dropout = True
        self.nclass = nclass
        if not (temperature > 0):
            raise ValueError(
                "Error: temperature should always be larger than 0. Provided: %s"
                % str(temperature)
            )
        self.temperature = temperature
        self.layers()

    def layers(self):
        model = [
            nn.Conv3d(1, self.C, kernel_size=self.first_kern, padding=1),
            self.norm_layer(self.C, affine=True),
            nn.ReLU(True),
        ]

        for i in range(self.n_downsampling):
            mult = 2**i
            model += [
                nn.Conv3d(
                    self.C * mult, self.C * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                self.norm_layer(self.C * mult * 2, affine=True),
                nn.ReLU(True),
            ]

        mult = 2**self.n_downsampling
        for i in range(self.n_blocks):
            model += [
                ResnetBlock3D(
                    self.C * mult,
                    "zero",
                    norm_layer=self.norm_layer,
                    use_dropout=self.use_dropout,
                )
            ]

        for i in range(self.n_downsampling):
            mult = 2 ** (self.n_downsampling - i)
            model += [
                nn.ConvTranspose3d(
                    self.C * mult,
                    int(self.C * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                self.norm_layer(int(self.C * mult / 2), affine=True),
                nn.ReLU(True),
            ]

        self.model = nn.Sequential(*model)

        self.do1_s = nn.Dropout3d(p=0.5)
        self.fconv1_s = nn.Sequential(
            nn.Conv3d(self.C, self.C, kernel_size=1), nn.ReLU(True)
        )  # added this non-linearity
        self.do2_s = nn.Dropout3d(p=0.5)

        self.do1_u = nn.Dropout3d(p=0.5)
        self.fconv1_u = nn.Sequential(
            nn.Conv3d(self.C, self.C, kernel_size=1), nn.ReLU(True)
        )  # added this non-linearity
        self.do2_u = nn.Dropout3d(p=0.5)

        self.output_segm = nn.Conv3d(self.C, self.nclass, kernel_size=1)
        self.output_uncertainty = nn.Conv3d(self.C, self.nclass, kernel_size=1)

    def forward(self, input):
        x = self.model(input)

        # segmentation
        x_segm = self.do1_s(x)
        x_segm = self.fconv1_s(x_segm)
        x_segm = self.do2_s(x_segm)
        x_logits = self.output_segm(x_segm)
        x_softmax = softmax_temperature(x_logits, self.temperature)

        # uncertainty prediction
        x_uncertainty = self.do1_u(x)
        x_uncertainty = self.fconv1_u(x_uncertainty)
        x_uncertainty = self.do2_u(x_uncertainty)
        x_uncertainty = self.output_uncertainty(x_uncertainty)

        return x_softmax, x_logits, x_uncertainty

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params


if __name__ == "__main__":
    # #2D dilated network
    network = DilatedNetwork2D(nclass=5, temperature=1, receptive_field=35)
    network = network.to("cuda")
    print(network)
    params = list(network.parameters())
    print(network.count_parameters(trainable=True))
    input = torch.randn(40, 1, 35, 35).to("cuda")
    out_softmax, out_logits = network(input)
    print(out_softmax.shape, out_logits.shape)

    # #Testing softmax temperature
    # out_logits = [0.1, 0.2, 6]
    # out_logits = np.asarray(out_logits, dtype=float)
    # out_logits = np.reshape(out_logits, (1,3,1,1))
    # out_logits = torch.from_numpy(out_logits)
    # out_softmax = softmax_temperature(out_logits, temperature=1)
    # print("temperature: ", 1)
    # print(out_softmax[0,0], out_logits[0,0])
    # print(out_softmax[0,1], out_logits[0,1])
    # print(out_softmax[0,2], out_logits[0,2])
    #
    # out_softmax = softmax_temperature(out_logits, temperature=40)
    # print("temperature: ", 40)
    # print(out_softmax[0,0], out_logits[0,0])
    # print(out_softmax[0,1], out_logits[0,1])
    # print(out_softmax[0,2], out_logits[0,2])
    #
    # out_softmax = softmax_temperature(out_logits, temperature=100)
    # print("temperature: ", 100)
    # print(out_softmax[0,0], out_logits[0,0])
    # print(out_softmax[0,1], out_logits[0,1])
    # print(out_softmax[0,2], out_logits[0,2])

    # # 3D Unet
    # network = UNet3DSmall(nclass=3, temperature=1)
    # network = network.to("cuda")
    # print(network)
    # params = list(network.parameters())
    # print(network.count_parameters(trainable=True))
    # n=72 #moet een meervoud van 8 zijn
    # input = torch.randn(1, 1,n,n,n).to('cuda')
    # out_softmax, out_logits = network(input)
    # print(out_softmax.shape, out_logits.shape)

    # 3D Resnet
    # network = Resnet2D(nclass=3, temperature=1, architecture="Resnet2DNEW")
    # network = network.to("cuda")
    # print(network)
    # params = list(network.parameters())
    # print(network.count_parameters(trainable=True))
    # n=64 #moet een meervoud van 8 zijn
    # input = torch.randn(1, 1,n,n).to('cuda')
    # out_softmax, out_logits = network(input)
    # print(out_softmax.shape, out_logits.shape)

    # #3D Resnet
    # network = Resnet3DUncertainty(nclass=3, temperature=1)
    # network = network.to("cuda")
    # print(network)
    # params = list(network.parameters())
    # print(network.count_parameters(trainable=True))
    # n=64 #moet een meervoud van 8 zijn
    # input = torch.randn(1, 1,n,n,n).to('cuda')
    # out_softmax, out_logits, out_uncertainty = network(input)
    # print(out_softmax.shape, out_logits.shape)

    # # 3D Unet_Strided
    # network = UNet3D_stridedConv(nclass=3, temperature=1)
    # network = network.to("cuda")
    # print(network)
    # params = list(network.parameters())
    # print(network.count_parameters(trainable=True))
    # n=80 #moet een meervoud van 8 zijn
    # input = torch.randn(15, 1,n,n,n).to('cuda')
    # out_softmax, out_logits = network(input)
    # print(out_softmax.shape, out_logits.shape)
