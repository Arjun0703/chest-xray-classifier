import torch.nn as nn
import torchxrayvision as xrv

NUM_OUTPUTS = 18


def build_densenet():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.op_threshs = None
    return model


def build_simple_cnn():
    def conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        conv_block(1, 32),
        conv_block(32, 64),
        conv_block(64, 128),
        conv_block(128, 256),
        conv_block(256, 512),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_OUTPUTS),
    )


def build_model(name):
    if name == "densenet":
        return build_densenet()
    if name == "simple":
        return build_simple_cnn()
    raise ValueError(f"Unknown model: {name!r}")
