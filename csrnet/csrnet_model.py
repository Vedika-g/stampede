import torch
import torch.nn as nn


# ============================================================
# CSRNet - Frontend / Backend Layer Builder
# ============================================================

def make_layers(cfg, in_channels=3, dilation=False):

    layers = []

    dilation_rate = 2 if dilation else 1

    for value in cfg:

        if value == "M":

            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

        else:

            layers.append(
                nn.Conv2d(
                    in_channels,
                    value,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate
                )
            )

            layers.append(
                nn.ReLU(inplace=True)
            )

            in_channels = value

    return nn.Sequential(*layers)


# ============================================================
# CSRNet
# ============================================================

class CSRNet(nn.Module):

    def __init__(self):

        super(CSRNet, self).__init__()

        # ----------------------------------------------------
        # Frontend
        # VGG-style convolutional layers
        # ----------------------------------------------------

        self.frontend = make_layers(
            [
                64,
                64,
                "M",

                128,
                128,
                "M",

                256,
                256,
                256,
                "M",

                512,
                512,
                512
            ]
        )

        # ----------------------------------------------------
        # Backend
        # Dilated convolution layers
        # ----------------------------------------------------

        self.backend = make_layers(
            [
                512,
                512,
                512,
                256,
                128,
                64
            ],
            in_channels=512,
            dilation=True
        )

        # ----------------------------------------------------
        # Output layer
        # Produces a single-channel density map
        # ----------------------------------------------------

        self.output_layer = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )


    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, x):

        x = self.frontend(x)

        x = self.backend(x)

        x = self.output_layer(x)

        return x