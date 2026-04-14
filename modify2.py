
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


# ==========================================
# 1. 基础组件：MLP 与 前沿特征增强 (LKA)
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


class LKA(nn.Module):
    """
    大核注意力机制 (Large Kernel Attention)
    来源: Visual Attention Network (VAN)
    优势: 结合了局部上下文、长距离依赖和通道适应性，比普通的 SE 通道注意力强很多。
    """

    def __init__(self, dim):
        super().__init__()
        # 1. 提取局部空间特征 (5x5 Depth-wise)
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # 2. 提取长距离空间特征 (5x5 Dilated Depth-wise, dilation=3 相当于 13x13 感受野)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=6, groups=dim, dilation=3)
        # 3. 通道混合 (1x1 Conv)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class Modern_FEB_LKA(nn.Module):
    """使用 LKA 替代普通 Channel Attention 的特征增强模块"""

    def __init__(self, in_channels):
        super(Modern_FEB_LKA, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),  # GELU 在现代网络中表现通常优于 LeakyReLU
        )
        self.lka = LKA(in_channels)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        feat = self.local_conv(x)
        feat = self.lka(feat)  # 经过大核注意力提取长距离依赖
        feat = self.proj(feat)
        return feat + identity


class SpatialAttention(nn.Module):
    """空间注意力机制，用于 MLP 融合后的边缘细节恢复"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度计算最大值和平均值，突出边缘特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        return x * self.sigmoid(scale)


# ==========================================
# 2. 网络主体：HANet + 增强型 SegFormer Decoder
# ==========================================

class HANet_MLP_Final(nn.Module):
    def __init__(self, embed_dim=256):
        super(HANet_MLP_Final, self).__init__()
        # Encoder: DenseNet121
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features
        self.ch0, self.ch2, self.ch3, self.ch_b = 64, 256, 512, 1024

        # 引入前沿的 LKA 特征增强
        self.feb2 = Modern_FEB_LKA(self.ch2)
        self.feb3 = Modern_FEB_LKA(self.ch3)

        # MLP 投影层
        self.linear0 = MLP(self.ch0, embed_dim)
        self.linear2 = MLP(self.ch2, embed_dim)
        self.linear3 = MLP(self.ch3, embed_dim)
        self.linear_b = MLP(self.ch_b, embed_dim)

        # 融合层
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # 增加空间边缘增强模块
        self.spatial_refine = SpatialAttention(kernel_size=7)

        # 最终预测
        self.final_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)

        # 辅助输出头 (用于 Deep Supervision)
        self.aux0 = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.aux2 = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.aux3 = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x):
        # 1. Encoder
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x)))
        e2 = self.encoder.denseblock1(self.encoder.pool0(e0))
        e3 = self.encoder.denseblock2(self.encoder.transition1(e2))
        e4 = self.encoder.denseblock3(self.encoder.transition2(e3))
        b = self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e4)))

        # 2. LKA 特征增强
        s2 = self.feb2(e2)
        s3 = self.feb3(e3)

        # 3. All-MLP Decoder 过程
        target_size = e0.size()[2:]
        _l0 = self.linear0(e0)
        _l2 = F.interpolate(self.linear2(s2), size=target_size, mode='bilinear', align_corners=False)
        _l3 = F.interpolate(self.linear3(s3), size=target_size, mode='bilinear', align_corners=False)
        _lb = F.interpolate(self.linear_b(b), size=target_size, mode='bilinear', align_corners=False)

        # 4. 融合与空间细化
        fused = self.linear_fuse(torch.cat([_l0, _l2, _l3, _lb], dim=1))
        fused = self.spatial_refine(fused)  # <-- 新增：强化融合后的边界特征

        # 5. 输出
        out = torch.sigmoid(
            F.interpolate(self.final_conv(fused), size=x.size()[2:], mode='bilinear', align_corners=False))

        if self.training:
            out0 = torch.sigmoid(F.interpolate(self.aux0(_l0), size=x.size()[2:], mode='bilinear', align_corners=False))
            out2 = torch.sigmoid(F.interpolate(self.aux2(_l2), size=x.size()[2:], mode='bilinear', align_corners=False))
            out3 = torch.sigmoid(F.interpolate(self.aux3(_lb), size=x.size()[2:], mode='bilinear', align_corners=False))
            return out, out0, out2, out3
        else:
            return out


class HybridLoss(nn.Module):
    def __init__(self, weight_focal=0.5, weight_dice=0.5):
        super(HybridLoss, self).__init__()
        self.focal = FocalLoss()  # 替换掉了原有的 BCE
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        smooth = 1e-5
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_score = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        dice_loss = 1.0 - dice_score
        return (self.weight_focal * focal_loss) + (self.weight_dice * dice_loss.mean())


# 其他计算指标代码和 main 函数保持原样即可...
# ==========================================
# 4. 混合损失函数与评估指标
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

    return {
        "dice": dice.tolist(), "iou": iou.tolist(), "acc": acc.tolist(),
        "pre": pre.tolist(), "rec": rec.tolist()
    }


# ==========================================
# 5. 数据集与加载器
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
# 6. 主控台
# ==========================================
def main():
    MODE = "train"

    save_path = "best_model_base.pth"
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | ⚙️ Mode: {MODE}")

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

    model = HANet_MLP_Final().to(device)

    if MODE == "train":
        train_loader = DataLoader(BUSIDataset(all_imgs[:train_size], all_masks[:train_size], True), batch_size=8,
                                  shuffle=True)
        val_loader = DataLoader(
            BUSIDataset(all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size], False),
            batch_size=8, shuffle=False)

        criterion = HybridLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        num_epochs = 15
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Training Progress (No Wavelet Baseline)')
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

                if isinstance(outputs, tuple):
                    out, out1, out2, out3 = outputs
                    loss_main = criterion(out, masks)
                    target_size = masks.shape[2:]
                    out1_up = F.interpolate(out1, size=target_size, mode='bilinear', align_corners=True)
                    out2_up = F.interpolate(out2, size=target_size, mode='bilinear', align_corners=True)
                    out3_up = F.interpolate(out3, size=target_size, mode='bilinear', align_corners=True)
                    loss_aux1 = criterion(out1_up, masks)
                    loss_aux2 = criterion(out2_up, masks)
                    loss_aux3 = criterion(out3_up, masks)
                    loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2 + 0.4 * loss_aux3
                else:
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
        plt.savefig('training_curve_no_wavelet.png', dpi=150)

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
    print("\n🏆 mynet_Baseline  最终测试集成绩 🏆")
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