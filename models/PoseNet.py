import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels,n1x1,kernel_size=1),
            nn.ReLU()
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return y


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()

        self.key = key
        # TODO: Define loss headers
        # LossHeader 1
        self.loss_header1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.ReLU(),
            init('loss1/conv', nn.Conv2d(512,128,kernel_size=1,stride=1), weights),
            nn.ReLU(),
            nn.Flatten(),
            init('loss1/fc', nn.Linear(2048,1024), weights),
            nn.Dropout(p=0.7),
        )
        
        # LossHeader 2
        self.loss_header2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.ReLU(),
            init('loss2/conv', nn.Conv2d(528,128,kernel_size=1,stride=1), weights),
            nn.ReLU(),
            nn.Flatten(),
            init('loss2/fc', nn.Linear(2048,1024), weights),
            nn.Dropout(p=0.7),
        )
        # LossHeader 1 & 2 outputs
        self.fc_h12_out1 = nn.Linear(1024,3)
        self.fc_h12_out2 = nn.Linear(1024,4)

        # LossHeader 3
        self.loss_header3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024,2048),
            nn.Dropout(p=0.4),
        )
        self.fc_h3_out1 = nn.Linear(2048,3)
        self.fc_h3_out2 = nn.Linear(2048,4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        if self.key == 1:
            x = self.loss_header1(x)
            xyz = self.fc_h12_out1(x)
            wpqr = self.fc_h12_out2(x)
        if self.key == 2:
            x = self.loss_header2(x)
            xyz = self.fc_h12_out1(x)
            wpqr = self.fc_h12_out2(x)
        if self.key == 3:
            x = self.loss_header3(x)
            xyz = self.fc_h3_out1(x)
            wpqr = self.fc_h3_out2(x)
        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        
        self.pre_layers = nn.Sequential(
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1), weights),
            nn.ReLU(),
            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights)

        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)

        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.relu = nn.ReLU()

        # LossHeader initialization
        self.loss1 = LossHeader(key=1)
        self.loss2 = LossHeader(key=2)
        self.loss3 = LossHeader(key=3)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        x = self.pre_layers(x)
        x = self._3a(x)
        x = self._3b(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self._4a(x)
        loss1_xyz, loss1_wpqr = self.loss1(x)

        x = self._4b(x)
        x = self._4c(x)
        x = self._4d(x)
        loss2_xyz, loss2_wpqr = self.loss2(x)

        x = self._4e(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self._5a(x)
        x = self._5b(x)
        loss3_xyz, loss3_wpqr = self.loss3(x) 
        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        x_gt = poseGT[:,:3]
        w_gt = poseGT[:,3:]
        w_gt_norm = F.normalize(w_gt, p=2)
        # print('poseGT', poseGT.shape)
        # print('p1_xyz', p1_xyz.shape)
        # print('p1_wpqr', p1_wpqr.shape)
        # print('w_gt',w_gt)
        # print('w_gt_norm',w_gt_norm)

        loss1_x = torch.sqrt(torch.mean(torch.square(F.pairwise_distance(x_gt, p1_xyz))))
        loss1_w = torch.sqrt(torch.mean(torch.square(F.pairwise_distance(w_gt_norm, p1_wpqr)))) * self.w1_wpqr

        loss2_x = torch.sqrt(torch.mean(torch.square(F.pairwise_distance(x_gt, p2_xyz))))
        loss2_w = torch.sqrt(torch.mean(torch.square(F.pairwise_distance(w_gt_norm, p2_wpqr)))) * self.w2_wpqr

        loss3_x = torch.sqrt(torch.mean(torch.square(F.pairwise_distance(x_gt, p3_xyz))))
        loss3_w = torch.sqrt(torch.mean(torch.square(F.pairwise_distance(w_gt_norm, p3_wpqr)))) * self.w3_wpqr
        # print('l1',loss1_x,loss1_w)
        # print('l2',loss2_x,loss2_w)
        # print('l3',loss3_x,loss3_w)
        
        loss = self.w1_xyz*(loss1_x+loss1_w) + self.w2_xyz*(loss2_x+loss2_w) + self.w3_xyz*(loss3_x+loss3_w)
        return loss