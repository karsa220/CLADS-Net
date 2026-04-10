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

import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    """标准的双层卷积解码块"""

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


class DoubleLevelFusion(nn.Module):
    """
    简化版的 DLF (Double-Level Fusion) 模块
    用于交叉融合 CNN 的局部特征和 Transformer 的全局特征
    """

    def __init__(self, cnn_dim, vit_dim, out_dim):
        super(DoubleLevelFusion, self).__init__()
        # 统一通道数
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.vit_proj = nn.Sequential(
            nn.Conv2d(vit_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        # 空间注意力门控机制 (模拟 Cross-Attention)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, out_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, vit_feat):
        # 统一维度
        c_feat = self.cnn_proj(cnn_feat)
        v_feat = self.vit_proj(vit_feat)

        # 拼接并融合
        concat_feat = torch.cat([c_feat, v_feat], dim=1)
        fused = self.fusion_conv(concat_feat)

        # 通道注意力重新校准
        att_weight = self.channel_att(fused)
        out = fused * att_weight + c_feat  # 残差连接偏向于保留局部细节
        return out


class HiFormer(nn.Module):
    """
    高度还原 HiFormer 核心结构的变体网络：
    1. CNN 分支: ResNet50 (提取层级局部特征)
    2. Transformer 分支: ViT 架构 (提取深层全局特征)
    3. DLF 融合: 在深层将两路特征进行交叉融合
    """

    def __init__(self, img_dim=256, in_channels=3, out_channels=1, embed_dim=512):
        super(HiFormer, self).__init__()

        # ==================================
        # 1. CNN 分支 (基于 ResNet50)
        # ==================================
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # (B, 64, 128, 128)
        self.pool = resnet.maxpool

        self.cnn_layer1 = resnet.layer1  # (B, 256, 64, 64)
        self.cnn_layer2 = resnet.layer2  # (B, 512, 32, 32)
        self.cnn_layer3 = resnet.layer3  # (B, 1024, 16, 16)

        # ==================================
        # 2. Transformer 分支 (轻量化 ViT)
        # ==================================
        # 直接从原图提取 Patch (16x16 降采样)
        self.patch_size = 16
        num_patches = (img_dim // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # ==================================
        # 3. 交叉融合层 (Double-Level Fusion)
        # ==================================
        # 将 CNN layer2 (32x32) 和 Transformer特征 融合
        self.vit_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dlf_1 = DoubleLevelFusion(cnn_dim=512, vit_dim=embed_dim, out_dim=256)

        # 将 CNN layer3 (16x16) 和 Transformer特征 融合
        self.dlf_2 = DoubleLevelFusion(cnn_dim=1024, vit_dim=embed_dim, out_dim=512)

        # ==================================
        # 4. 解码器 Decoder
        # ==================================
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(512 + 256, 256)  # 融合 DLF2(512) 和 DLF1(256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(256 + 256, 128)  # 融合 Dec4(256) 和 cnn_layer1(256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64)  # 融合 Dec3(128) 和 stem(64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(64, 32)  # 恢复到原图分辨率 256x256

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape

        # --- 1. CNN 分支提取 ---
        e0 = self.stem(x)  # (B, 64, 128, 128)
        e1 = self.pool(e0)
        e1 = self.cnn_layer1(e1)  # (B, 256, 64, 64)
        e2 = self.cnn_layer2(e1)  # (B, 512, 32, 32)
        e3 = self.cnn_layer3(e2)  # (B, 1024, 16, 16)

        # --- 2. Transformer 分支提取 ---
        v_patch = self.patch_embed(x)  # (B, 512, 16, 16)
        v_flat = v_patch.flatten(2).transpose(1, 2)  # (B, 256, 512)
        v_flat = v_flat + self.pos_embed
        v_trans = self.transformer(v_flat)  # (B, 256, 512)

        # 恢复为 2D 拓扑结构
        v_feat = v_trans.transpose(1, 2).reshape(B, -1, H // 16, W // 16)  # (B, 512, 16, 16)

        # --- 3. Double-Level Fusion 交叉融合 ---
        # 融合底层: CNN layer3 (16x16) + ViT feat (16x16) -> (B, 512, 16, 16)
        feat_dlf2 = self.dlf_2(e3, v_feat)

        # 融合高层: CNN layer2 (32x32) + ViT feat (上采样至 32x32) -> (B, 256, 32, 32)
        v_feat_up = self.vit_up1(v_feat)
        feat_dlf1 = self.dlf_1(e2, v_feat_up)

        # --- 4. 级联解码 ---
        d4 = self.up4(feat_dlf2)  # (B, 512, 32, 32)
        d4 = self.dec4(torch.cat([d4, feat_dlf1], dim=1))  # (B, 256, 32, 32)

        d3 = self.up3(d4)  # (B, 256, 64, 64)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))  # 加入 CNN 浅层细节 (B, 128, 64, 64)

        d2 = self.up2(d3)  # (B, 128, 128, 128)
        d2 = self.dec2(torch.cat([d2, e0], dim=1))  # (B, 64, 128, 128)

        d1 = self.up1(d2)  # (B, 64, 256, 256)
        d1 = self.dec1(d1)  # (B, 32, 256, 256)

        out = torch.sigmoid(self.final_conv(d1))
        return out


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
    model = HiFormer().to(device)
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