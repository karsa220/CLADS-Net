import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt


class LayerNorm2d(nn.Module):
    """ConvNeXt 强烈推荐使用的 2D LayerNorm，比 BatchNorm 在小 BatchSize 下表现更好"""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class WTConv2d(nn.Module):
    """在小波域进行大核深度可分离卷积的核心算子"""

    def __init__(self, in_channels, wt_type='db2'):
        super().__init__()
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=True)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=True)

        # 在低频(LL)分支使用 5x5 的深度可分离卷积，等效于在原图上有 > 10x10 的超大感受野
        self.wt_dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)

    def forward(self, x):
        curr_shape = x.shape
        pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
        x_padded = F.pad(x, pads) if sum(pads) > 0 else x

        wt_out = wavelet_transform(x_padded, self.wt_filter)
        ll, hl, lh, hh = wt_out[:, :, 0], wt_out[:, :, 1], wt_out[:, :, 2], wt_out[:, :, 3]

        # 仅对低频全局信息进行大核空间聚合
        ll_conv = self.wt_dwconv(ll)

        wt_out_modified = torch.stack([ll_conv, hl, lh, hh], dim=2)
        out = inverse_wavelet_transform(wt_out_modified, self.iwt_filter)
        return out[:, :, :curr_shape[2], :curr_shape[3]]


class WTConvNeXtBlock(nn.Module):
    """
    完整的 WTConvNeXt Block
    架构：WTConv(Depthwise) -> LayerNorm -> 1x1 Conv(扩维) -> GELU -> 1x1 Conv(降维) -> LayerScale -> 残差
    """

    def __init__(self, dim, wt_type='db2', drop_path=0.):
        super().__init__()
        self.dwconv = WTConv2d(dim, wt_type=wt_type)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        # ConvNeXt 的倒置瓶颈设计 (Inverted Bottleneck)，中间维度扩大 4 倍
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # LayerScale 技巧：初始化为极小值，利于深层网络稳定训练
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-6)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return input + x

# ==========================================
# 1. 小波变换基础算子 (优化版：引入反射填充避免边缘伪影)
# ==========================================
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad_h = filters.shape[2] // 2 - 1
    pad_w = filters.shape[3] // 2 - 1

    # 🌟 优化点 1: 使用反射填充 (Reflection Padding) 替代零填充，极大减少高频人造伪影
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    x = F.conv2d(x, filters, stride=2, groups=c)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad_h = filters.shape[2] // 2 - 1
    pad_w = filters.shape[3] // 2 - 1
    x = x.reshape(b, c * 4, h_half, w_half)
    # 逆变换利用标准反卷积还原即可
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=(pad_h, pad_w))
    return x


# ==========================================
# 2. 核心模块：带高频门控与LL大核增强的 SFEB (终极优化版)
# ==========================================
class SFEB_Advanced(nn.Module):
    def __init__(self, in_channels, wt_type='db2'):
        super(SFEB_Advanced, self).__init__()

        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=True)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=True)

        self.ll_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.hf_dir_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]).view(1, 3, 1, 1))

        self.hf_mix = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 3),
            nn.ReLU(inplace=True)
        )
        self.hf_spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(in_channels // 4, 1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        local_feat = self.local_conv(x)

        curr_shape = local_feat.shape
        pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
        local_feat_padded = F.pad(local_feat, pads) if sum(pads) > 0 else local_feat

        wt_out = wavelet_transform(local_feat_padded, self.wt_filter)
        ll = wt_out[:, :, 0, :, :]
        hl = wt_out[:, :, 1, :, :]
        lh = wt_out[:, :, 2, :, :]
        hh = wt_out[:, :, 3, :, :]

        ll_enhanced = self.ll_enhance(ll)

        hf_stack = torch.stack([hl, lh, hh], dim=1)
        hf_stack = hf_stack * self.hf_dir_weights.unsqueeze(-1)

        b, _, c, h_f, w_f = hf_stack.shape
        hf_cat = hf_stack.view(b, 3 * c, h_f, w_f)

        hf_mixed = self.hf_mix(hf_cat)

        hf_avg = torch.mean(hf_mixed, dim=1, keepdim=True)
        hf_max, _ = torch.max(hf_mixed, dim=1, keepdim=True)
        hf_mask_in = torch.cat([hf_avg, hf_max], dim=1)
        hf_mask = self.hf_spatial_attention(hf_mask_in)

        hf_gated = hf_mixed * hf_mask
        hl_g, lh_g, hh_g = torch.chunk(hf_gated, 3, dim=1)

        wt_out_modified = torch.stack([ll_enhanced, hl_g, lh_g, hh_g], dim=2)
        freq_feat = inverse_wavelet_transform(wt_out_modified, self.iwt_filter)
        freq_feat = freq_feat[:, :, :curr_shape[2], :curr_shape[3]]
        freq_feat = freq_feat + local_feat

        fused_feat = self.fusion_conv(freq_feat)
        att_weight = self.channel_att(self.gap(fused_feat))
        out = fused_feat * att_weight

        return out + identity


# ==========================================
# 3. 注意力卷积块 (保持不变)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class BottleneckSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(BottleneckSelfAttention, self).__init__()
        # 为了减少计算量，Query 和 Key 的通道数降维到 1/8
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 🌟 核心技巧：可学习的缩放参数 gamma
        # 初始化为 0，意味着在训练初期，这个模块等同于什么都不做 (纯残差 x)
        # 随着训练进行，网络会自己学习需要多少全局注意力，这样不会破坏预训练权重
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. 计算 Query 和 Key [B, C//8, H*W]
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)  # [B, H*W, C//8]
        proj_key = self.key_conv(x).view(b, -1, h * w)  # [B, C//8, H*W]

        # 2. 计算注意力权重矩阵 [B, H*W, H*W]
        # 除以 sqrt(d_k) 防止 softmax 梯度消失
        energy = torch.bmm(proj_query, proj_key) / ((c // 8) ** 0.5)
        attention = F.softmax(energy, dim=-1)

        # 3. 将注意力施加到 Value 上 [B, C, H*W]
        proj_value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, H*W]

        # 4. 恢复形状并加上残差
        out = out.view(b, c, h, w)
        out = self.gamma * out + x

        return out
class AttentionConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
# ==========================================
# 4. 网络主体
# ==========================================
# ==========================================
# 4. 网络主体 (优化版：SFEB仅应用于中层)
# ==========================================
class HANet_Final(nn.Module):
    def __init__(self):
        super(HANet_Final, self).__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 256, 512, 1024

        self.sfeb2 = SFEB_Advanced(self.ch2, wt_type='db2')
        self.sfeb3 = SFEB_Advanced(self.ch3, wt_type='db2')

        # === 🌟 改进：使用两个堆叠的 WTConvNeXt Block 替换传统的 Self-Attention ===
        # 两个堆叠模块能提供深不可测的感受野，同时参数量远小于标准注意力
        self.bottleneck_wtconvnext = nn.Sequential(
            WTConvNeXtBlock(dim=1024, wt_type='db2'),
            WTConvNeXtBlock(dim=1024, wt_type='db2')
        )

        self.ag1 = AttentionGate(F_g=128, F_l=self.ch1, F_int=64)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = AttentionConvBlock(1024 + self.ch4, 512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = AttentionConvBlock(512 + self.ch3, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = AttentionConvBlock(256 + self.ch2, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = AttentionConvBlock(128 + self.ch1, 64)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec0 = AttentionConvBlock(64, 32)

        self.out3 = nn.Conv2d(256, 1, kernel_size=1)
        self.out2 = nn.Conv2d(128, 1, kernel_size=1)
        self.out1 = nn.Conv2d(64, 1, kernel_size=1)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x)))
        e2 = self.encoder.denseblock1(self.encoder.pool0(e0))
        e3 = self.encoder.denseblock2(self.encoder.transition1(e2))
        e4 = self.encoder.denseblock3(self.encoder.transition2(e3))
        b = self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e4)))

        # === 🌟 改进：在这里通过 WTConvNeXt 提取全局上下文 ===
        b = self.bottleneck_wtconvnext(b)

        # 跳跃连接准备
        skip1 = e0
        skip2 = self.sfeb2(e2)
        skip3 = self.sfeb3(e3)
        skip4 = e4

        # Decoder 融合
        d4 = self.dec4(torch.cat([self.up4(b), skip4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))

        up_d2 = self.up1(d2)
        skip1_gated = self.ag1(g=up_d2, x=skip1)
        d1 = self.dec1(torch.cat([up_d2, skip1_gated], dim=1))

        out = torch.sigmoid(self.final_conv(self.dec0(self.up0(d1))))

        if self.training:
            out1 = torch.sigmoid(self.out1(d1))
            out2 = torch.sigmoid(self.out2(d2))
            out3 = torch.sigmoid(self.out3(d3))
            return out, out1, out2, out3
        else:
            return out

import torch
import torch.nn as nn
import torch.nn.functional as F



# ==========================================
# 5. 混合损失函数与 🌟增强版评估指标🌟
# ==========================================
import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice  # 变量名改为 weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5

        # 展平预测和标签
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # 计算交集
        intersection = (pred_flat * target_flat).sum(dim=1)

        # 计算 Dice 系数
        # 公式: (2 * Intersection) / (Pred_Sum + Target_Sum)
        dice_score = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)

        # Dice Loss 是 1 减去 Dice 系数
        dice_loss = 1.0 - dice_score

        # 返回加权混合损失
        return (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss.mean())

def calculate_metrics(pred, target, smooth=1e-5):
    """计算 ACC, PRECISION, RECALL, IOU, DICE 五大核心指标"""
    pred_bin = (pred > 0.5).float()
    target_bin = target.float()

    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target_bin.view(target_bin.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)
    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)

    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    acc = (tp + tn + smooth) / (tp + fp + fn + tn + smooth)
    pre = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)

    return {
        "dice": dice.tolist(),
        "iou": iou.tolist(),
        "acc": acc.tolist(),
        "pre": pre.tolist(),
        "rec": rec.tolist()
    }


# ==========================================
# 6. 数据集与加载器
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return TF.to_tensor(image), TF.to_tensor(mask)


def get_dataset_paths(data_dir):
    image_paths, mask_paths = [], []
    categories = ['benign', 'malignant']
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir): continue
        for file_name in os.listdir(cat_dir):
            if file_name.endswith('.png') and not file_name.endswith('_mask.png'):
                img_path = os.path.join(cat_dir, file_name)
                mask_path = os.path.join(cat_dir, file_name.replace('.png', '_mask.png'))
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
    return image_paths, mask_paths


# ==========================================
# 7. 主控台
# ==========================================
def main():
    # 🌟🌟🌟 控制面板 🌟🌟🌟
    MODE = "train"  # 改为 "test" 则跳过训练，直接加载保存好的权重评估

    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    save_path = "best_model.pth"  # 权重保存的文件名

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | ⚙️ Mode: {MODE}")

    # 1. 准备数据 (使用统一的 random seed)
    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ 未找到数据！请检查路径。")
        return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # 初始化模型
    model = HANet_Final().to(device)

    # ==========================
    # 模式 A：训练模式
    # ==========================
    if MODE == "train":
        train_loader = DataLoader(BUSIDataset(all_imgs[:train_size], all_masks[:train_size], True), batch_size=8,
                                  shuffle=True)
        val_loader = DataLoader(
            BUSIDataset(all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size], False),
            batch_size=8, shuffle=False)

        criterion = HybridLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        num_epochs = 35
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0

        # 画图初始化
        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Training Progress (HANet_Final)')
        line_loss, = ax1.plot([], [], color='tab:red', marker='o', label='Train Loss')
        line_dice, = ax2.plot([], [], color='tab:blue', marker='s', label='Val Dice')
        ax1.legend([line_loss, line_dice], ['Train Loss', 'Val Dice'], loc='center right')
        history_loss, history_val_dice, epochs_list = [], [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0

            pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Train", leave=False)
            for images, masks in pbar_train:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                # 新的代码
                outputs = model(images)

                # 判断如果 outputs 是元组（说明在深度监督的训练阶段）
                if isinstance(outputs, tuple):
                    out, out1, out2, out3 = outputs

                    # 1. 计算主输出的 Loss (权重最大)
                    loss_main = criterion(out, masks)

                    # 2. 将低分辨率的辅助输出上采样 (Upsample) 到真实标签 masks 的尺寸
                    target_size = masks.shape[2:]  # 获取 (H, W)
                    out1_up = F.interpolate(out1, size=target_size, mode='bilinear', align_corners=True)
                    out2_up = F.interpolate(out2, size=target_size, mode='bilinear', align_corners=True)
                    out3_up = F.interpolate(out3, size=target_size, mode='bilinear', align_corners=True)

                    # 3. 计算辅助 Loss
                    loss_aux1 = criterion(out1_up, masks)
                    loss_aux2 = criterion(out2_up, masks)
                    loss_aux3 = criterion(out3_up, masks)

                    # 4. 加权求和 (通常主路给 1.0，辅助支路给 0.3 到 0.5)
                    loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2 + 0.4 * loss_aux3
                else:
                    # 验证(val)或测试(test)阶段，模型只返回一个输出
                    loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    metrics = calculate_metrics(outputs, masks)
                    val_dices.extend(metrics["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            print(
                f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # 动态更新图表
            epochs_list.append(epoch + 1)
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)
            line_loss.set_data(epochs_list, history_loss)
            line_dice.set_data(epochs_list, history_val_dice)
            ax1.set_xlim(1, max(2, epoch + 1))
            ax1.set_ylim(min(history_loss) * 0.9, max(history_loss) * 1.1)
            ax2.set_ylim(min(history_val_dice) * 0.9, max(1.0, max(history_val_dice) * 1.1))
            fig.canvas.draw()
            fig.canvas.flush_events()

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Model Saved] Val Dice: {best_val_dice:.4f}")



        plt.ioff()
        plt.savefig('training_curve_HANet.png', dpi=150)

    # ==========================
    # 模式 B：纯测试 / 评估模式
    # ==========================
    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的【测试集 (Test Set)】上评估最终性能...")

    test_loader = DataLoader(BUSIDataset(all_imgs[train_size + val_size:], all_masks[train_size + val_size:], False),
                             batch_size=8, shuffle=False)

    if not os.path.exists(save_path):
        print(f"❌ 找不到权重文件: {save_path}！请先运行 train 模式训练。")
        return

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
    # 🌟 GPU 预热 (Warm-up)
    # 防止第一次启动推理时的寻址延迟拉低整体速度评估
    print("🔥 正在进行 GPU 预热 (Warm-up)...")
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 初始化时间累加器
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)

            # ---- 计时开始 ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            outputs = model(images)

            # ---- 计时结束 ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            # 累加耗时和样本数
            total_infer_time += (end_time - start_time)
            total_samples += images.size(0)

            # 记录精度指标
            metrics = calculate_metrics(outputs, masks)
            for k in test_res.keys():
                test_res[k].extend(metrics[k])

    # 🌟 计算平均推理时间和 FPS
    avg_time_per_image = (total_infer_time / total_samples) * 1000  # 转为毫秒 ms
    fps = 1.0 / (total_infer_time / total_samples)  # 帧率 FPS
    print("\n🏆 mynet_Final 最终测试集成绩 🏆")
    print(f"🔹 Dice     : {np.mean(test_res['dice']):.4f}")
    print(f"🔹 IoU      : {np.mean(test_res['iou']):.4f}")
    print(f"🔹 ACC      : {np.mean(test_res['acc']):.4f}")
    print(f"🔹 Precision: {np.mean(test_res['pre']):.4f}")
    print(f"🔹 Recall   : {np.mean(test_res['rec']):.4f}")
    print("-" * 50)
    print(f"⚡ 推理速度评估 (Device: {device}) ⚡")
    print(f"⏱️ 平均耗时 : {avg_time_per_image:.2f} ms / image")
    print(f"🚀 F P S    : {fps:.2f} frames / second")
    print("=" * 50)


if __name__ == "__main__":
    main()