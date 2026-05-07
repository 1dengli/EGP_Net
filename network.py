import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile
from Res2Net_U.model.Res2Net_v1b import res2net50_v1b_26w_4s
from z_models.models import GPP
from z_models.models import DAF
# 最终最终

class egp(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel):
        super(egp, self).__init__()

        self.con1 = nn.Conv2d(channel, 3, 1)
        # ---- Res2Net50 Backbone ----  
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)

        self.conv_boun_f = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_boun_c1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.boun_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.conv_boun_c2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.boun_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.edge_out = nn.Conv2d(32, 1, 1)


        self.g2 = GPG.GPG_2([1024, 2048], 1024, up_kwargs = {'mode': 'bilinear', 'align_corners': True})
        self.g3 = GPG.GPG_3([512, 1024, 2048], 512, up_kwargs = {'mode': 'bilinear', 'align_corners': True})
        self.g4 = GPG.GPG_4([256, 512, 1024, 2048], 256, up_kwargs = {'mode': 'bilinear', 'align_corners': True})

        self.espcn1 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        self.espcn2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

        self.espcn3 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 2, 3, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 1),
        )
        self.af = DAF.DAF(1, 1)


    def forward(self, x):
        x_0 = self.con1(x)
        x_1 = self.conv_boun_f(x_0)

        x = self.res2net.conv1(x_0)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x) # 64*64*64
        x_pool = self.res2net.maxpool(x)

        x1 = self.res2net.layer1(x) # 256*64*64

        x2 = self.res2net.layer2(x1) # 512*32*32

        x3 = self.res2net.layer3(x2) # 1024*16*16

        x4 = self.res2net.layer4(x3) # 2048*8*8

        x_cat = torch.cat([x_1, x_pool], dim=1)
        x_b_1 = self.conv_boun_c1(x_cat)
        x_boun_up = self.boun_up(x_b_1)
        x_b_2 = self.conv_boun_c2(x_boun_up)
        x_boun_up2 = self.boun_up2(x_b_2)
        x_local_out = self.edge_out(x_boun_up2)

        g2 = self.g2(x3, x4)
        e1 = self.espcn3(g2)

        g3 = self.g3(x2, x3, x4)
        e2 = self.espcn2(g3)

        g4 = self.g4(x1, x2, x3, x4)
        e3 = self.espcn1(g4)

        out_global_out = e1 + e2 + e3

        out = self.af(x_local_out, out_global_out)

        return out




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ras = egp(1).to(device)
    print(summary(ras, (1, 128, 128)))

    input = torch.randn(1, 1, 128, 128).to(device)
    flops, params = profile(ras, inputs=(input,))
    print(f"FLOPs: {flops / 1e9} G")