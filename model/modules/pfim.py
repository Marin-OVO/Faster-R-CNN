import torch
import torch.nn as nn


def standard_normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))


class PFIM(nn.Module):
    def __init__(self, ich=256, hch=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ich, hch, 3, padding=1),
            nn.BatchNorm2d(hch),
            nn.ReLU(inplace=True),

            nn.Conv2d(hch, hch, 3, padding=1),
            nn.BatchNorm2d(hch),
            nn.ReLU(inplace=True),

            nn.Conv2d(hch, hch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hch, hch, 3, padding=1),
        )

        self.mu_head = nn.Conv2d(hch, ich, kernel_size=1, bias=True)
        self.sigma_head = nn.Sequential(
            nn.Conv2d(hch, ich, kernel_size=1, bias=True),
            nn.Softplus()
        )

        nn.init.kaiming_normal_(self.mu_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.kaiming_normal_(self.sigma_head[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.sigma_head[0].bias, -3.0)

    def forward(self, y):
        y = torch.clamp(y, 0.0, 1.0)

        x = self.conv(y)

        mu = self.mu_head(x)
        mu = mu.clamp(min=-10, max=10)

        sigma = self.sigma_head(x)
        sigma = sigma.clamp(min=0.01, max=3.0)

        # 添加噪声
        noise = torch.empty_like(y).uniform_(-0.5, 0.5)
        y_q = y + noise

        # 计算信息熵损失
        a = ((y_q + 0.5 - mu) / (sigma + 1e-3)).clamp(min=-6, max=6)
        b = ((y_q - 0.5 - mu) / (sigma + 1e-3)).clamp(min=-6, max=6)

        p = standard_normal_cdf(a) - standard_normal_cdf(b)
        p = p.clamp(min=1e-6)

        Ry = -torch.log2(p)
        Ry = Ry.clamp(max=12.0)

        L_IE = Ry.mean()

        return {
            'mu': mu,
            'sigma': sigma,
            'L_IE': L_IE,
        }