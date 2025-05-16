from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import torch.nn as nn


class EfficientNetB4Classifier:
    def __init__(self, device, num_classes):
        weights_b4 = EfficientNet_B4_Weights.DEFAULT
        self.pretrained_efficient_net_b4 = efficientnet_b4(weights=weights_b4).to(device)

        for parameter in self.pretrained_efficient_net_b4.parameters():
            parameter.requires_grad = False

        for parameter in self.pretrained_efficient_net_b4.features[-1].parameters():
            parameter.requires_grad = True

        for m in self.pretrained_efficient_net_b4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)

        self.pretrained_efficient_net_b4.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes).to(device)

