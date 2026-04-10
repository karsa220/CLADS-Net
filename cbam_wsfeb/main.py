import os
import random
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
import gc
import pywt


# ==========================================
# 1. 小波变换基础算子
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
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# ==========================================
# 2. 核心模块：带高频门控与LL大核增强的 SFEB (默认 db2)
# ==========================================
class SFEB_Advanced(nn.Module):
    def __init__(self, in_channels, wt_type='db2'):  # 🌟 改进：默认使用平滑性更好的 db2
        super(SFEB_Advanced, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.ll_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
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
        ll, hl, lh, hh = wt_out[:, :, 0], wt_out[:, :, 1], wt_out[:, :, 2], wt_out[:, :, 3]

        ll_enhanced = self.ll_enhance(ll)
        hf_cat = torch.cat([hl, lh, hh], dim=1)
        hf_mixed = self.hf_mix(hf_cat)

        hf_avg = torch.mean(hf_mixed, dim=1, keepdim=True)
        hf_max, _ = torch.max(hf_mixed, dim=1, keepdim=True)
        hf_mask = self.hf_spatial_attention(torch.cat([hf_avg, hf_max], dim=1))

        hf_gated = hf_mixed * hf_mask
        hl_g, lh_g, hh_g = torch.chunk(hf_gated, 3, dim=1)

        wt_out_modified = torch.stack([ll_enhanced, hl_g, lh_g, hh_g], dim=2)
        freq_feat = inverse_wavelet_transform(wt_out_modified, self.iwt_filter)
        freq_feat = freq_feat[:, :, :curr_shape[2], :curr_shape[3]]

        freq_feat = freq_feat + local_feat
        fused_feat = self.fusion_conv(freq_feat)
        att_weight = self.channel_att(self.gap(fused_feat))

        return (fused_feat * att_weight) + identity


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


# ==========================================
# 4. 网络主体
# ==========================================
class HANet_Final(nn.Module):
    def __init__(self):
        super(HANet_Final, self).__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 256, 512, 1024

        self.sfeb1 = SFEB_Advanced(self.ch1, wt_type='db2')
        self.sfeb2 = SFEB_Advanced(self.ch2, wt_type='db2')
        self.sfeb3 = SFEB_Advanced(self.ch3, wt_type='db2')
        self.sfeb4 = SFEB_Advanced(self.ch4, wt_type='db2')

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

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x)))
        e2 = self.encoder.denseblock1(self.encoder.pool0(e0))
        e3 = self.encoder.denseblock2(self.encoder.transition1(e2))
        e4 = self.encoder.denseblock3(self.encoder.transition2(e3))
        b = self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e4)))

        skip1 = self.sfeb1(e0)
        skip2 = self.sfeb2(e2)
        skip3 = self.sfeb3(e3)
        skip4 = self.sfeb4(e4)

        d4 = self.dec4(torch.cat([self.up4(b), skip4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))
        out = torch.sigmoid(self.final_conv(self.dec0(self.up0(d1))))
        return out


# ==========================================
# 5. 混合损失函数与指标计算
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5

        # 🌟 改进：按样本级别计算交并比，避免被 Batch 内极端样本带偏
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        jaccard_loss = 1.0 - (intersection + smooth) / (union + smooth)
        return bce_loss + jaccard_loss.mean()


def calculate_metrics(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice.tolist(), iou.tolist()


# ==========================================
# 6. 数据集与增强加载
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train  # 🌟 改进：隔离训练集与验证集的增强逻辑

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 统一尺寸
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        # 仅在训练阶段启用随机数据增强防过拟合
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 随机旋转 -15 到 15 度
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
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 🌟 改进：先获取所有路径，手动打乱并拆分，再实例化 Dataset
    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ 未找到数据！请检查路径。")
        return

    # 打乱数据
    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)

    # 拆分路径列表
    train_imgs, train_masks = all_imgs[:train_size], all_masks[:train_size]
    val_imgs, val_masks = all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size]
    test_imgs, test_masks = all_imgs[train_size + val_size:], all_masks[train_size + val_size:]

    # 实例化 Dataset（只有 Train 开启数据增强）
    train_dataset = BUSIDataset(train_imgs, train_masks, is_train=True)
    val_dataset = BUSIDataset(val_imgs, val_masks, is_train=False)
    test_dataset = BUSIDataset(test_imgs, test_masks, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"✅ 数据加载完毕 | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = HANet_Final().to(device)
    criterion = HybridLoss()

    # 🌟 改进：增加权重衰减并更换为余弦退火策略
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_dice = 0.0

    # ... (绘图初始化代码保持不变) ...
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax2.set_ylabel('Validation Dice', color='tab:blue')
    plt.title('Training Progress')
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
            outputs = model(images)
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
                dices, _ = calculate_metrics(outputs, masks)
                val_dices.extend(dices)

        avg_val_dice = np.mean(val_dices)

        # 🌟 改进：余弦退火每个 epoch 强制调度一次
        scheduler.step()

        print(
            f"📉 Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

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
            torch.save(model.state_dict(), "best_model.pth")
            print(f"🌟 [New Best Model Saved] Val Dice: {best_val_dice:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    plt.ioff()
    plt.savefig('training_curve.png', dpi=150)
    print("📈 训练曲线已保存为 training_curve.png")

    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的【测试集 (Test Set)】上评估最终性能...")

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_dices, test_ious = [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dices, ious = calculate_metrics(outputs, masks)
            test_dices.extend(dices)
            test_ious.extend(ious)

    print("\n🏆 最终测试集成绩 🏆")
    print(f"Test Dice: {np.mean(test_dices):.4f}")
    print(f"Test IoU:  {np.mean(test_ious):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()