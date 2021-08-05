import torch.nn as nn
import math

###### VGG16 #############
class VGG(nn.Module):
    def __init__(self, features): # 완성된 모델을 넘겨받음. list형태로
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential( # 딥러닝 모델을 연결해주는 함수. Sequential함수는 리스트의 시작주소를 파라미터로 받는다.
            nn.Dropout(), # 결국 이런 layer들이 list의 원소 형태로 저장되는 것.
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x): # x는 최종 값을 저장할 변수
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg: # v = output channel
        if v == 'M': # 만약 MaxPooling을 진행해야하는 경우라면
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm: #batch normalization을 수행하는 경우
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16():
    # cfg shows 'kernel size'
    # 'M' means 'max pooling'
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(make_layers(cfg))