import torch
import torch.nn.functional as F
import torch.nn as nn


def Y_Upper(y1, y2, hyp_prm=1.3):
    """
    根据两幅亮度图构造目标亮度
    """
    C = 0.0001

    # 全局均值
    y1_mean = torch.mean(y1)
    y2_mean = torch.mean(y2)

    # 去均值
    y1_mean_sub = y1 - y1_mean
    y2_mean_sub = y2 - y2_mean

    # 对比度幅值
    c1 = torch.norm(y1_mean_sub)
    c2 = torch.norm(y2_mean_sub)
    c_upperArrow = torch.maximum(c1, c2)

    # 放大系数
    c_upperArrow = c_upperArrow * hyp_prm

    # 单位结构向量
    s1 = y1_mean_sub / (c1 + C)
    s2 = y2_mean_sub / (c2 + C)

    # 融合结构
    s_upperDash = s1 + s2
    s_upperArrow = s_upperDash / (torch.norm(s_upperDash) + C)

    # 最终目标
    y_upperArrow = c_upperArrow * s_upperArrow

    return y_upperArrow


def _th_fspecial_gauss(size, sigma, device, dtype):
    """
    用 torch 直接在目标 device 上生成高斯核
    输出形状: [1, 1, size, size]
    """
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def th_SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    """
    结构相似性损失
    img1, img2: [B, 1, H, W]
    """
    window = _th_fspecial_gauss(
        size=size,
        sigma=sigma,
        device=img1.device,
        dtype=img1.dtype
    )

    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = F.conv2d(img1, window)
    mu2 = F.conv2d(img2, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window) - mu1_mu2

    value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    value = value.mean()

    return 1 - value


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()

    def forward(self, ir_image, visimage_bri, res_weight, fus_img, upper_weight, mean_weight):
        target = Y_Upper(ir_image, visimage_bri, upper_weight).to(ir_image.device)
        ssim_loss = th_SSIM_LOSS(target, fus_img)
        pre_loss = torch.abs(torch.mean(mean_weight - torch.sum(res_weight, dim=1, keepdim=False)))
        return ssim_loss + pre_loss, [ssim_loss, pre_loss, pre_loss]