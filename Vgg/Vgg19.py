import torch
import torch.nn as nn
from torchvision.models import vgg19
from collections import namedtuple
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained = vgg19(pretrained=True)
        self.features = vgg_pretrained.features
        self.avgpool = vgg_pretrained.avgpool
        self.classifier = vgg_pretrained.classifier
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.cnn_struct = [
            'conv1_1','conv1_2','pool1',
            'conv2_1','conv2_2','pool2',
            'conv3_1','conv3_2','conv3_3','conv3_4','pool3',
            'conv4_1','conv4_2','conv4_3','conv4_4','pool4',
            'conv5_1','conv5_2','conv5_3','conv5_4','pool5'
        ]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x[:, [2, 1, 0], :, :]
        # 减去均值
        features = []
        x -= torch.tensor(self.VGG_MEAN).to(x.device).view(1, 3, 1, 1)
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in {'1', '6', '11', '20', '22', '29'}:
                features.append(x)
    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        self.logits = x
        softmax_output = nn.functional.softmax(x, dim=1)
    
        vgg_outputs = namedtuple("VggOutputs", ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1','softmax'])
        return vgg_outputs(*features, softmax_output)