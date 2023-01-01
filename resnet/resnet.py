import torch
import torch.nn as nn

class ResNetBlock(nn.Module):

    '''
    base convolutional block of resnet
    '''

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        identity_downsample=None,
        stride=1
    ):
        super(ResNetBlock, self).__init__()
        self.expansion_factor = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion_factor, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion_factor)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, X):
        identity = X

        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # reshape identity appropriately, apply skip connection
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        out += identity
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, ResNetBlock, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        '''
        inputs: 
        ResNetBlock: convolutional block
        layers: list number of times to include the conv block
        image_channels: number of input image channels
        num_classes: number of classification classes
        '''
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self._make_layer(ResNetBlock, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(ResNetBlock, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(ResNetBlock, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(ResNetBlock, layers[3], out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*4, num_classes)
    
    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out

    def _make_layer(self, ResNetBlock, num_residule_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # downsample identity
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )
        
        layers.append(ResNetBlock(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residule_blocks - 1):
            layers.append(ResNetBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(ResNetBlock, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
   return ResNet(ResNetBlock, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
   return ResNet(ResNetBlock, [3, 8, 36, 3], img_channels, num_classes)

def test():
    net = ResNet50()
    X = torch.randn(2, 3, 224, 224)
    y = net(X).to('cpu')
    print(y.shape)

if __name__ == "__main__":
    test()