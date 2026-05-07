import torch
import torch.nn.functional as F
import torch.nn as nn



class GAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=1):
        super(GAG, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(x1 + g1))
        out = x * psi
        out += x
        return out


class SimpleAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(SimpleAttention, self).__init__()

        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # 输入的 x 为 (B, C, H, W)
        batch_size, channels, height, width = x.size()
        # 平均池化操作来获取全局上下文信息
        gap = F.adaptive_avg_pool2d(x, (1, 1))  # (B, C, 1, 1)
        gap = gap.view(batch_size, channels)  # (B, C)
        # 通过全连接层得到注意力权重
        attention = F.relu(self.fc1(gap))
        attention = torch.sigmoid(self.fc2(attention))  # (B, C)
        # 将注意力权重应用于输入特征
        attention = attention.view(batch_size, channels, 1, 1)  # (B, C, 1, 1)
        # x = x * attention  # 对特征进行加权

        return attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DHAF(nn.Module):
    def __init__(self, c1, c2):
        super(DHAF, self).__init__()
        self.se1 = SimpleAttention(c1)
        self.se2 = SimpleAttention(c2)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.act = nn.Sigmoid()
        self.SA = SpatialAttention(kernel_size=3)  # 替换为膨胀卷积
        self.conv = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )
        # self.transformer = TransformerBlock(dim=c2, num_heads=4)  # 加入 Transformer

    def forward(self, x1, x2):
        weight_x1 = self.se1(x1)
        weight_x2 = self.se2(x2)
        weight_all = self.act(self.GAP(x1 + x2))

        alpha_x1 = weight_x1 / (weight_x1 + weight_x2 + 1e-6)
        alpha_x2 = weight_x2 / (weight_x1 + weight_x2 + 1e-6)
        alpha_x1 = alpha_x1 * weight_all
        alpha_x2 = alpha_x2 * weight_all
        X = alpha_x1 * x1 + alpha_x2 * x2
        X = self.conv(X)*self.SA(X)
        # X =   # 进一步增强特征
        # X = self.conv(X) + self.SA(X)
        return X


class DAF(nn.Module):
    def __init__(self,in_dim_g,indim_l, is_bottom=False):
        super(DAF, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_dim_g,indim_l,kernel_size=1,stride=1),
                                nn.BatchNorm2d(indim_l),
                                nn.ReLU())
        self.GAG = GAG(indim_l, indim_l, indim_l)
        self.DHAF= DHAF(indim_l,indim_l)

    def forward(self,x1,x2):
        ##x1_>high  x2->lower
        x1=self.conv(x1)
        x2=self.conv(x2)
        xeag=self.GAG(x1,x2)
        xeag=x1+xeag
        x_mtr=self.DHAF(x1,x2)
        x_mtr=x2+x_mtr
        # X=torch.cat((xeag,x_mtr),dim=1)
        X = xeag+x_mtr
        return X



if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')

    f_d = torch.randn(1, 1, 32, 32).to(device)
    f_s = torch.randn(1, 1, 32, 32).to(device)

    model = DAF(1, 1).to(device)

    y = model(f_d, f_s)

    print("输入特征维度：", f_d.shape)
    print("输入特征维度：", f_s.shape)
    print("输出特征维度：", y.shape)