import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from torch.nn import Parameter
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """

        return self.backbone(x)

class EfficientNet4Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """

        return self.backbone(x)

class MultiPredictModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = self.backbone = EfficientNet.from_pretrained('efficientnet-b0', num_classes=8)
    def forward(self, x):
        return self.backbone(x)

class MultiPredict4Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = self.backbone = EfficientNet.from_pretrained('efficientnet-b4', num_classes=8)
    def forward(self, x):
        return self.backbone(x)

class ArcMarginModel(nn.Module):
    def __init__(self, num_classes, margin_m, margin_s, easy_margin=True):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, 512))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class ViTModel(nn.Module):
    def __init__(self,
        num_classes,
        img_size=384,
        multi_drop=False,
        multi_drop_rate=0.5,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_drop = multi_drop

        self.model = timm.create_model(
            'vit_base_patch16_384', pretrained=True
        )
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()

        self.head = nn.Linear(n_features, num_classes)
        self.head_drops = nn.ModuleList()
        for i in range(5):
            self.head_drops.append(nn.Dropout(multi_drop_rate))

    def forward(self, x):
        h = self.model(x)

        if self.multi_drop:
            for i, dropout in enumerate(self.head_drops):
                if i == 0:
                    output = self.head(dropout(h))
                else:
                    output += self.head(dropout(h))
            output /= len(self.head_drops)
        else:
            output = self.head(h)
        return output

class MultiPredictViTModel(nn.Module):
    def __init__(self,
        num_classes,
        img_size=384,
        multi_drop=False,
        multi_drop_rate=0.5,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_drop = multi_drop

        self.model = timm.create_model(
            'vit_base_patch16_384', pretrained=True
        )
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()

        self.head = nn.Linear(n_features, 8)
        self.head_drops = nn.ModuleList()
        for i in range(5):
            self.head_drops.append(nn.Dropout(multi_drop_rate))

    def forward(self, x):
        h = self.model(x)

        if self.multi_drop:
            for i, dropout in enumerate(self.head_drops):
                if i == 0:
                    output = self.head(dropout(h))
                else:
                    output += self.head(dropout(h))
            output /= len(self.head_drops)
        else:
            output = self.head(h)
        return output

class Testload(nn.Module):
    def __init__(self,
        num_classes,
        img_size=384,
        multi_drop=False,
        multi_drop_rate=0.5,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_drop = multi_drop

        self.model = MultiPredictViTModel(num_classes=8)

        self.model.load_state_dict(torch.load('/opt/ml/pycharm/src/model/conf_31_vit16_customloss_adamp1e-5_noval_multipred2/4_22.13%.pth', map_location='cuda'))

    def forward(self, x):
        h = self.model(x)
        return h

class Testload2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EfficientNet4Model(18)
        self.backbone.load_state_dict(torch.load('/opt/ml/pycharm/src/model/conf_28_32_efb4_customloss_adamp1e-5_noval/9_93.21%.pth', map_location='cuda'))
    def forward(self, x):
        return self.backbone(x)