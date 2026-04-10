import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc





class ConvBlock(nn.Module):
    """标准的双层卷积用于解码器"""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SwinUNet_Hybrid(nn.Module):
    """
    基于 PyTorch 官方 Swin-T 的 UNet 变体：
    Encoder: Swin Transformer (Tiny)
    Decoder: CNN + Skip Connections
    """

    def __init__(self, out_channels=1):
        super(SwinUNet_Hybrid, self).__init__()
        # 加载官方预训练的 Swin-T (如果网络不好，会自动下载，稍等片刻即可)
        swin_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        self.features = swin_model.features

        # Swin-T 的阶段通道数分别是: 96, 192, 384, 768
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(768 + 384, 384)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(384 + 192, 192)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(192 + 96, 96)

        # 最后一层恢复到原始分辨率 (Swin 最初有 4x 下采样)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(96, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 记录每层的特征用于 Skip Connection
        skips = []
        out = x

        # Swin-T features 包含 8 个模块：
        # 0: Patch Partition (B, 96, H/4, W/4)
        # 1: Swin Stage 1
        # 2: Patch Merging (B, 192, H/8, W/8)
        # 3: Swin Stage 2
        # 4: Patch Merging (B, 384, H/16, W/16)
        # 5: Swin Stage 3
        # 6: Patch Merging (B, 768, H/32, W/32)
        # 7: Swin Stage 4

        for i, module in enumerate(self.features):
            out = module(out)
            # 提取 4 个层级的特征 (经过转置恢复成 [B, C, H, W] 格式)
            if i in [1, 3, 5, 7]:
                # Swin 输出形状是 [B, H, W, C]，需要转换为 [B, C, H, W]
                feat = out.permute(0, 3, 1, 2).contiguous()
                skips.append(feat)

        s1, s2, s3, bottleneck = skips[0], skips[1], skips[2], skips[3]

        # --- Decoder ---
        d4 = self.dec4(torch.cat([self.up4(bottleneck), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s1], dim=1))
        d1 = self.dec1(self.up1(d2))

        return torch.sigmoid(self.final_conv(d1))


# ==========================================
# 3. 混合损失函数与指标计算 (完全保留)
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_jaccard=0.6):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_jaccard = weight_jaccard

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        jaccard_loss = 1.0 - (intersection + smooth) / (union + smooth)

        return (self.weight_bce * bce_loss) + (self.weight_jaccard * jaccard_loss.mean())


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
# 4. 数据集与增强加载 (完全保留)
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
# 5. 主控台
# ==========================================
def main():
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

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

    train_imgs, train_masks = all_imgs[:train_size], all_masks[:train_size]
    val_imgs, val_masks = all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size]
    test_imgs, test_masks = all_imgs[train_size + val_size:], all_masks[train_size + val_size:]

    train_dataset = BUSIDataset(train_imgs, train_masks, is_train=True)
    val_dataset = BUSIDataset(val_imgs, val_masks, is_train=False)
    test_dataset = BUSIDataset(test_imgs, test_masks, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"✅ 数据加载完毕 | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # 🌟 修改点：在此处实例化你的 Baseline (TransUNet)
    model = SwinUNet_Hybrid().to(device)
    criterion = HybridLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_dice = 0.0

    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax2.set_ylabel('Validation Dice', color='tab:blue')
    plt.title('Baseline Training Progress (TransUNet)')
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
            torch.save(model.state_dict(), "best_baseline_transunet.pth")
            print(f"🌟 [New Best Baseline Saved] Val Dice: {best_val_dice:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    plt.ioff()
    plt.savefig('baseline_training_curve.png', dpi=150)
    print("📈 基线训练曲线已保存为 baseline_training_curve.png")

    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的【测试集 (Test Set)】上评估 Baseline 最终性能...")

    model.load_state_dict(torch.load("best_baseline_transunet.pth"))
    model.eval()
    test_dices, test_ious = [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dices, ious = calculate_metrics(outputs, masks)
            test_dices.extend(dices)
            test_ious.extend(ious)

    print("\n🏆 Baseline 最终测试集成绩 (TransUNet) 🏆")
    print(f"Test Dice: {np.mean(test_dices):.4f}")
    print(f"Test IoU:  {np.mean(test_ious):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()