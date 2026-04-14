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
from tqdm import tqdm


# ==========================================
# 1. 基础组件：MLP 与 特征增强
# ==========================================

class MLP(nn.Module):
    """借鉴 SegFormer 的线性层，用于统一通道维度"""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


class Basic_FEB_Leaky(nn.Module):
    """带 LeakyReLU 的特征增强模块"""

    def __init__(self, in_channels):
        super(Basic_FEB_Leaky, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(in_channels // 4, 1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        feat = self.local_conv(x)
        att_weight = self.channel_att(self.gap(feat))
        return feat * att_weight + identity


# ==========================================
# 2. 上采样平滑模块 (方案一核心)
# ==========================================

class ProgressiveUpsample(nn.Module):
    """
    针对 16 倍极度放大的渐进式上采样模块 (8x8 -> 128x128)
    分两步：插值到1/4尺寸 -> 卷积平滑 -> 插值到目标尺寸 -> 卷积平滑
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.smooth1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        mid_size = (target_size[0] // 4, target_size[1] // 4)
        x = F.interpolate(x, size=mid_size, mode='bilinear', align_corners=False)
        x = self.smooth1(x)

        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.smooth2(x)
        return x


class SingleUpsample(nn.Module):
    """
    针对 2 倍或 4 倍放大的普通平滑上采样模块
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.smooth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.smooth(x)


# ==========================================
# 3. 网络主体：HANet 渐进式并行融合
# ==========================================

class HANet_Progressive_Final(nn.Module):
    def __init__(self, embed_dim=256):
        super(HANet_Progressive_Final, self).__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features

        self.ch0, self.ch2, self.ch3, self.ch_b = 64, 256, 512, 1024

        self.feb2 = Basic_FEB_Leaky(self.ch2)
        self.feb3 = Basic_FEB_Leaky(self.ch3)

        self.linear0 = MLP(self.ch0, embed_dim)
        self.linear2 = MLP(self.ch2, embed_dim)
        self.linear3 = MLP(self.ch3, embed_dim)
        self.linear_b = MLP(self.ch_b, embed_dim)

        # --- 核心引入：带平滑卷积的上采样模块 ---
        self.up_2 = SingleUpsample(embed_dim)  # 64x64 -> 128x128
        self.up_3 = SingleUpsample(embed_dim)  # 32x32 -> 128x128
        self.up_b = ProgressiveUpsample(embed_dim)  # 8x8 -> 128x128 (两步法)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.final_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)

        # 辅助输出头
        self.aux0 = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.aux2 = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.aux3 = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x)))  # [B, 64, 128, 128]
        e2 = self.encoder.denseblock1(self.encoder.pool0(e0))  # [B, 256, 64, 64]
        e3 = self.encoder.denseblock2(self.encoder.transition1(e2))  # [B, 512, 32, 32]

        e4 = self.encoder.denseblock3(self.encoder.transition2(e3))  # [B, 1024, 16, 16]
        b = self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e4)))  # [B, 1024, 8, 8]

        s2 = self.feb2(e2)
        s3 = self.feb3(e3)

        target_size = e0.size()[2:]  # 128x128

        # --- 使用平滑上采样替代直接插值 ---
        _l0 = self.linear0(e0)  # 原尺寸
        _l2 = self.up_2(self.linear2(s2), target_size)  # 平滑上采样 2 倍
        _l3 = self.up_3(self.linear3(s3), target_size)  # 平滑上采样 4 倍
        _lb = self.up_b(self.linear_b(b), target_size)  # 渐进平滑上采样 16 倍

        # 并行拼接融合
        fused = self.linear_fuse(torch.cat([_l0, _l2, _l3, _lb], dim=1))

        out = torch.sigmoid(
            F.interpolate(self.final_conv(fused), size=x.size()[2:], mode='bilinear', align_corners=False))

        if self.training:
            out0 = torch.sigmoid(F.interpolate(self.aux0(_l0), size=x.size()[2:], mode='bilinear', align_corners=False))
            out2 = torch.sigmoid(F.interpolate(self.aux2(_l2), size=x.size()[2:], mode='bilinear', align_corners=False))
            out3 = torch.sigmoid(F.interpolate(self.aux3(_lb), size=x.size()[2:], mode='bilinear', align_corners=False))
            return out, out0, out2, out3
        else:
            return out


# ==========================================
# 4. 损失函数与评价指标
# ==========================================

class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_score = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        dice_loss = 1.0 - dice_score
        return (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss.mean())


def calculate_metrics(pred_bin, target, smooth=1e-5):
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)
    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)

    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    acc = (tp + tn + smooth) / (tp + fp + fn + tn + smooth)
    pre = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    return {"dice": dice.tolist(), "iou": iou.tolist(), "acc": acc.tolist(), "pre": pre.tolist(), "rec": rec.tolist()}


# ==========================================
# 5. 数据加载与主流程
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


def main():
    MODE = "train"  # 改为 "test" 测试最终性能
    save_path = "best_model_progressive_decoder.pth"
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 📐 Architecture: HANet-Progressive (Scheme 1)")

    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0: return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # 实例化模型
    model = HANet_Progressive_Final(embed_dim=256).to(device)

    if MODE == "train":
        train_loader = DataLoader(BUSIDataset(all_imgs[:train_size], all_masks[:train_size], True), batch_size=8,
                                  shuffle=True)
        val_loader = DataLoader(
            BUSIDataset(all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size], False),
            batch_size=8)

        criterion = HybridLoss()

        # 差异化学习率
        encoder_params = list(map(id, model.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in encoder_params, model.parameters())
        optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': 1e-4},
            {'params': base_params, 'lr': 1e-3}
        ], weight_decay=1e-4)

        num_epochs = 50
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0
        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0

            # 动态衰减深层监督权重
            aux_weight = max(0.1, 0.4 * (1 - epoch / num_epochs))

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                out, out0, out2, out3 = model(images)
                loss_main = criterion(out, masks)
                loss_aux = (criterion(out0, masks) + criterion(out2, masks) + criterion(out3, masks)) / 3.0
                loss = loss_main + aux_weight * loss_aux

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'AuxWt': f"{aux_weight:.2f}"})

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    pred_bin = (outputs > 0.5).float()
                    val_dices.extend(calculate_metrics(pred_bin, masks)["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()
            print(
                f"📉 Epoch {epoch + 1} | Loss: {train_loss_epoch / len(train_loader):.4f} | Val Dice: {avg_val_dice:.4f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Saved Best Model: {best_val_dice:.4f}")

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

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # 直接通过阈值二值化（去除了 cv2 连通域后处理）
            pred_bin = (outputs > 0.5).float()

            metrics = calculate_metrics(pred_bin, masks)
            for k in test_res.keys():
                test_res[k].extend(metrics[k])

    print("\n🏆 Progressive_Baseline 最终测试集成绩 🏆")
    print(f"🔹 Dice     : {np.mean(test_res['dice']):.4f}")
    print(f"🔹 IoU      : {np.mean(test_res['iou']):.4f}")
    print(f"🔹 ACC      : {np.mean(test_res['acc']):.4f}")
    print(f"🔹 Precision: {np.mean(test_res['pre']):.4f}")
    print(f"🔹 Recall   : {np.mean(test_res['rec']):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()