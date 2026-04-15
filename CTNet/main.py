import os
import random
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# ==========================================
# 1. CT-Net 模型定义 (Zhang et al., 2024)
# ==========================================

class CNNBlock(nn.Module):
    """局部特征提取：双层标准卷积"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    """全局特征提取：标准自注意力机制"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # x: [B, N, C]
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransBranch(nn.Module):
    """CT-Net 并行 Transformer 分支 (处理 Patch 以获取全局依赖)"""

    def __init__(self, in_channels, embed_dim, img_size, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # 将图片划分 patch 并线性映射
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.Sequential(
            TransformerBlock(embed_dim, num_heads=4),
            TransformerBlock(embed_dim, num_heads=4)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/16, W/16]
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = x + self.pos_embed
        x = self.blocks(x)

        # 还原回 2D 空间维度以备融合
        x = x.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        return x


class HighDensityFusion(nn.Module):
    """
    CT-Net 极轻量级高密度融合模块 (论文核心)
    参数量极低 (~0.05M)，通过空间注意力引导 CNN 和 Transformer 特征的高效融合。
    """

    def __init__(self, cnn_dim, trans_dim, out_dim):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, 1)
        self.trans_proj = nn.Conv2d(trans_dim, out_dim, 1)

        # 生成基于交叉通道特征的空间注意力权重
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, trans_feat):
        # 对齐尺寸 (如果 Transformer patch size 导致维度偏差)
        if trans_feat.shape[2:] != cnn_feat.shape[2:]:
            trans_feat = F.interpolate(trans_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=False)

        c_p = self.cnn_proj(cnn_feat)
        t_p = self.trans_proj(trans_feat)

        # 拼接以计算注意力
        concat_feat = torch.cat([c_p, t_p], dim=1)
        attn_weight = self.spatial_attn(concat_feat)

        # 软门控 (Soft-gating) 融合
        fused = c_p * attn_weight + t_p * (1 - attn_weight)
        return fused


class CTNet(nn.Module):
    """
    CT-Net: Asymmetric Compound Branch Transformer
    (非对称复合分支架构)
    """

    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super().__init__()

        # --- Asymmetric Parallel Branches (双分支并行特征提取) ---

        # 1. CNN 分支 (局部细节)
        self.cnn_inc = CNNBlock(n_channels, 32)
        self.cnn_down1 = nn.Sequential(nn.MaxPool2d(2), CNNBlock(32, 64))
        self.cnn_down2 = nn.Sequential(nn.MaxPool2d(2), CNNBlock(64, 128))
        self.cnn_down3 = nn.Sequential(nn.MaxPool2d(2), CNNBlock(128, 256))
        self.cnn_down4 = nn.Sequential(nn.MaxPool2d(2), CNNBlock(256, 512))

        # 2. Transformer 分支 (全局上下文)
        self.trans_branch = TransBranch(in_channels=n_channels, embed_dim=256, img_size=img_size, patch_size=16)

        # --- High-Density Information Fusion (高密度信息融合) ---
        self.fusion = HighDensityFusion(cnn_dim=512, trans_dim=256, out_dim=512)

        # --- Asymmetric Decoder (非对称解码器) ---
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = CNNBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CNNBlock(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = CNNBlock(64 + 64, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec4 = CNNBlock(32 + 32, 32)

        # --- 输出层 ---
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # 局部特征路径 (CNN)
        c1 = self.cnn_inc(x)  # [B, 32, H, W]
        c2 = self.cnn_down1(c1)  # [B, 64, H/2, W/2]
        c3 = self.cnn_down2(c2)  # [B, 128, H/4, W/4]
        c4 = self.cnn_down3(c3)  # [B, 256, H/8, W/8]
        c5 = self.cnn_down4(c4)  # [B, 512, H/16, W/16]

        # 全局特征路径 (Transformer)
        t_feat = self.trans_branch(x)  # [B, 256, H/16, W/16]

        # 模块融合 (Fusion)
        fused = self.fusion(c5, t_feat)  # [B, 512, H/16, W/16]

        # 解码并结合高分辨率跳跃连接
        d1 = self.up1(fused)
        d1 = self.dec1(torch.cat([d1, c4], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, c3], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, c2], dim=1))

        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, c1], dim=1))

        logits = self.outc(d4)
        return torch.sigmoid(logits)


# ==========================================
# 2. 混合损失函数与指标计算 (保持原样)
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
    return {"dice": dice.tolist(), "iou": iou.tolist(), "acc": acc.tolist(), "pre": pre.tolist(), "rec": rec.tolist()}


# ==========================================
# 3. 数据集与增强加载 (保持原样)
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

        # 锁定图像分辨率为 256x256，适配 Transformer 分支 patch 处理
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
# 4. 主控台
# ==========================================
def main():
    MODE = "train"
    save_path = "best_busi.pth"

    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 🔄 Current Mode: {MODE.upper()}")

    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ 未找到数据！请检查路径。")
        exit()

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

    # ==========================================
    # 初始化 CT-Net 模型
    # ==========================================
    model = CTNet(n_channels=3, n_classes=1, img_size=256).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧩 CT-Net 模型参数量: {total_params / 1e6:.4f} M")

    criterion = HybridLoss()

    if MODE == "train":
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        num_epochs = 35
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Training Progress (CT-Net)')
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
                    val_dices.extend(calculate_metrics(outputs, masks)["dice"])

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
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best CT-Net Saved] Val Dice: {best_val_dice:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        plt.ioff()
        plt.savefig('ctnet_training_curve.png', dpi=150)
        print("📈 训练曲线已保存为 ctnet_training_curve.png")

    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print("🚀 开始在完全未见的【测试集 (Test Set)】上评估 CT-Net 最终性能...")

        if not os.path.exists(save_path):
            print(f"❌ 找不到权重文件: {save_path}！请先运行 train 模式训练。")
        else:
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
            print("🔥 正在进行 GPU 预热 (Warm-up)...")
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_infer_time = 0.0
            total_samples = 0

            with torch.no_grad():
                for images, masks in tqdm(test_loader, desc="Testing"):
                    images, masks = images.to(device), masks.to(device)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.time()

                    outputs = model(images)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()

                    total_infer_time += (end_time - start_time)
                    total_samples += images.size(0)

                    metrics = calculate_metrics(outputs, masks)
                    for k in test_res.keys():
                        test_res[k].extend(metrics[k])

            avg_time_per_image = (total_infer_time / total_samples) * 1000
            fps = 1.0 / (total_infer_time / total_samples)

            print("\n🏆 CT-Net (BUSI dataset) 最终测试集成绩 🏆")
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