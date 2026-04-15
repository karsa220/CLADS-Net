import os
import random
import time

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


# ==========================================
# 1. 基础卷积块 (Decoder 使用)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
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


# ==========================================
# 2. 网络主体：TransUNet (对比基线模型)
# ==========================================
class TransUNet(nn.Module):
    def __init__(self, img_dim=256, in_channels=3, out_channels=1, embed_dim=768, depth=8, num_heads=8):
        super(TransUNet, self).__init__()

        # 1. CNN Encoder (提取到 ResNet50 的 layer3)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 2. Patch Embedding (用 1x1 卷积将 ResNet 输出的通道数降维到 Transformer 的特征维度)
        self.patch_proj = nn.Conv2d(1024, embed_dim, kernel_size=1)

        # 3. Transformer Encoder (ViT Bottleneck)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 位置编码 (假设输入 256x256，经过 resnet layer3 后分辨率为 16x16)
        num_patches = (img_dim // 16) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # 4. Decoder 模块
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(embed_dim + 512, 512)  # Trans特征(768) + layer2跳跃(512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(512 + 256, 256)  # 上采样特征(512) + layer1跳跃(256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(256 + 64, 128)  # 上采样特征(256) + conv1跳跃(64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(128, 64)  # 恢复到原图分辨率 256x256

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)  # (B, 64, 128, 128)
        skip1 = x0

        x1 = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x1)  # (B, 256, 64, 64)
        skip2 = x1

        x2 = self.resnet.layer2(x1)  # (B, 512, 32, 32)
        skip3 = x2

        x3 = self.resnet.layer3(x2)  # (B, 1024, 16, 16) - 进入 Transformer

        # --- ViT Bottleneck ---
        B, C, H, W = x3.shape
        x_proj = self.patch_proj(x3)  # (B, 768, 16, 16)
        x_flat = x_proj.flatten(2).transpose(1, 2)  # (B, 256, 768)

        x_flat = x_flat + self.pos_embed  # 加入位置编码
        x_trans = self.transformer(x_flat)  # (B, 256, 768)

        x_trans = x_trans.transpose(1, 2).reshape(B, -1, H, W)  # 恢复为 (B, 768, 16, 16)

        # --- Decoder ---
        up4 = self.up4(x_trans)
        d4 = self.dec4(torch.cat([up4, skip3], dim=1))  # 输出: (B, 512, 32, 32)

        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat([up3, skip2], dim=1))  # 输出: (B, 256, 64, 64)

        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2, skip1], dim=1))  # 输出: (B, 128, 128, 128)

        up1 = self.up1(d2)
        d1 = self.dec1(up1)  # 输出: (B, 64, 256, 256)

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

    # ==========================================
    # 1. 运行模式设置
    # ==========================================
    MODE = "train"  # 可选: "train" 或 "test"
    save_path = "best_busi.pth"

    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 🔄 Current Mode: {MODE.upper()}")

    # ==========================================
    # 2. 数据集加载与划分
    # ==========================================
    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ 未找到数据！请检查路径。")
        # 如果在函数外直接 return 会报错，建议用 sys.exit() 或放在 main() 中
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
    # 3. 模型与损失函数初始化
    # ==========================================
    model = TransUNet().to(device)
    criterion = HybridLoss()

    # ==========================================
    # 4. 训练模式 (Train)
    # ==========================================
    if MODE == "train":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        num_epochs = 35
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0

        # 初始化绘图
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
                    val_dices.extend(calculate_metrics(outputs, masks)["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            print(
                f"📉 Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # 更新绘图数据
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

            # 保存最佳模型
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Baseline Saved] Val Dice: {best_val_dice:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        # 结束绘图
        plt.ioff()
        plt.savefig('baseline_training_curve.png', dpi=150)
        print("📈 基线训练曲线已保存为 baseline_training_curve.png")
    # ==========================================
    # 5. 测试模式 (Test) / 全指标评估
    # ==========================================
    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print("🚀 开始在完全未见的【测试集 (Test Set)】上评估 Baseline 最终性能...")

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
            print("\n🏆 unetplusplus (BUSI dataset)  最终测试集成绩 🏆")
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