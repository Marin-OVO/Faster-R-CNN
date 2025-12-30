import torch
import torch.nn as nn
import torch.nn.functional as F


def MGT(boxes, img_size, feature_size, device):
    """
    公式:
    f(p) = 1/N * Σ N(p|μ_box_i, Σ_box_i)
    th = 1/|S| * Σ N·f(p)
    MGT = (Sign(N·f(p) - th) + 1) × 0.25 + N·f(p)

    Args:
        boxes: Tensor [N, 4], 格式 [x1, y1, x2, y2]
        img_size: tuple (H_img, W_img) 原始图像尺寸
        feature_size: tuple (H_feat, W_feat) 特征图尺寸
        device: torch device

    Returns:
        MGT: Tensor [1, H_feat, W_feat]
    """
    H_img, W_img = img_size
    H_feat, W_feat = feature_size

    if isinstance(boxes, list):
        if len(boxes) == 0:
            return torch.zeros((1, H_feat, W_feat), device=device)
        boxes = boxes[0] if isinstance(boxes[0], torch.Tensor) else torch.tensor(boxes[0], device=device)

    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)

    N = len(boxes)
    if N == 0:
        return torch.zeros((1, H_feat, W_feat), device=device)

    scale_x = W_feat / W_img
    scale_y = H_feat / H_img
    boxes_feat = boxes.clone()
    boxes_feat[:, [0, 2]] *= scale_x
    boxes_feat[:, [1, 3]] *= scale_y

    y_coords = torch.arange(H_feat, device=device).float().unsqueeze(1).expand(H_feat, W_feat)
    x_coords = torch.arange(W_feat, device=device).float().unsqueeze(0).expand(H_feat, W_feat)
    grid = torch.stack([x_coords, y_coords], dim=-1)  # [H_feat, W_feat, 2]

    f_p = torch.zeros((H_feat, W_feat), device=device)

    for i in range(N):
        x1, y1, x2, y2 = boxes_feat[i]

        xi = (x1 + x2) / 2
        yi = (y1 + y2) / 2
        wi = x2 - x1
        hi = y2 - y1

        area_img = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])).item()

        if area_img < 64:  # 2-8 pixels (very tiny)
            alpha = 4
        elif area_img < 256:  # 8-16 pixels (tiny)
            alpha = 6
        elif area_img < 1024:  # 16-32 pixels (small)
            alpha = 8
        else:  # general objects
            alpha = 10

        mu = torch.tensor([xi, yi], device=device)

        sigma_x = (wi / alpha).clamp(min=1e-6)
        sigma_y = (hi / alpha).clamp(min=1e-6)

        diff = grid - mu  # [H_feat, W_feat, 2]

        exponent = -0.5 * ((diff[..., 0] / sigma_x) ** 2 + (diff[..., 1] / sigma_y) ** 2)
        gaussian = torch.exp(exponent)

        normalization = 1.0 / (2 * 3.14159265359 * sigma_x * sigma_y)

        f_p += normalization * gaussian

    f_p = f_p / N

    N_f_p = N * f_p
    th = N_f_p.mean()

    sign_term = torch.sign(N_f_p - th)
    MGT = (sign_term + 1) * 0.25 + N_f_p

    return MGT.unsqueeze(0)


class PGDP(nn.Module):
    def __init__(self, ich=256, hch=256):
        super().__init__()
        C = ich
        H = hch

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, H, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(C, H, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(C, H, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(C, H, 3, 1, 1)
        )

        self.pred = nn.Sequential(
            nn.Conv2d(H, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def LossPred(self, pred, target, th=0.5, pos_weight=10.0, neg_weight=0.1):
        if target is None:
            return torch.tensor(0.0, device=pred.device)

        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        weight = torch.where(target > th, pos_weight, neg_weight)
        loss = (weight * (pred - target) ** 2).mean()

        return loss

    def forward(self, p4, p3, p2, sigma=None, boxes=None, img_size=None):
        """
        Args:
            p4, p3, p2: FPN特征层
            sigma: PFIM输出的sigma图，shape [B, C, H, W]
            boxes: list of boxes or Tensor, 用于生成ground truth
            img_size: tuple (H, W) 原始图像尺寸
        """
        B, C, H2, W2 = p2.shape

        if sigma is None:
            sigma = torch.zeros_like(p2)
        else:
            sigma = torch.nan_to_num(sigma, nan=0.0, posinf=5.0, neginf=-5.0)
            sigma = sigma.clamp(min=-5.0, max=5.0)

        p2 = self.conv4(p2 + sigma)

        p3 = self.conv3(p3 + self.downsample(sigma))

        p4 = self.conv2(self.conv1(p4 + self.downsample(self.downsample(sigma))))

        p3 = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p2.shape[-2:], mode='bilinear', align_corners=False)

        Mpd2 = p2 + p3 + p4

        Mpd2 = torch.sigmoid(Mpd2)
        Mpd3 = torch.sigmoid(p3)
        Mpd4 = torch.sigmoid(p4)

        boxes_tensor = None
        if boxes is not None and img_size is not None:
            H_img, W_img = img_size
            boxes_tensor = boxes if isinstance(boxes, torch.Tensor) else boxes[0]

            mgt = MGT(boxes_tensor, (H_img, W_img), p2.shape[-2:], device=sigma.device)

            L_pred = (
                    self.LossPred(Mpd2, mgt) +
                    self.LossPred(Mpd3, mgt) +
                    self.LossPred(Mpd4, mgt)
            )
        else:
            mgt = None
            L_pred = torch.tensor(0.0, device=sigma.device)

        return {
            "Mpd2": Mpd2,
            "Mpd3": Mpd3,
            "Mpd4": Mpd4,
            "L_pred": L_pred,
            "mgt": mgt
        }