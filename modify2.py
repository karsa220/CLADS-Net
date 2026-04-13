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
# 1. 核心模块：基础特征增强模块
# ==========================================
class Basic_FEB(nn.Module):
    def __init__(self, in_channels):
        super(Basic_FEB, self).__init__()

        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

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

        # 1. 局部特征提取
        feat = self.local_conv(x)

        # 2. 通道注意力加权
        att_weight = self.channel_att(self.gap(feat))
        out = feat * att_weight

        # 3. 残差连接
        return out + identity


# ==========================================
# 2. 注意力卷积块
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
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, h * w)

        energy = torch.bmm(proj_query, proj_key) / ((c // 8) ** 0.5)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

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
# 3. 网络主体 (已修改：转置卷积 & 步长为2的卷积)
# ==========================================
class HANet_Final(nn.Module):
    def __init__(self):
        super(HANet_Final, self).__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features

        # 【修改 1】：将 DenseNet 原本的 pool0 (最大池化层) 替换为 步幅为 2 的可学习卷积层
        # 原本的 self.encoder.conv0 输出通道数为 64
        self.encoder.pool0 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 256, 512, 1024

        self.feb2 = Basic_FEB(self.ch2)
        self.feb3 = Basic_FEB(self.ch3)

        self.bottleneck_attn = BottleneckSelfAttention(in_channels=1024)
        self.ag1 = AttentionGate(F_g=128, F_l=self.ch1, F_int=64)

        # 【修改 2】：将 nn.Upsample 全部替换为 nn.ConvTranspose2d
        # scale_factor=2 对应 kernel_size=2, stride=2。设置前后通道数一致，交由 dec 融合降维
        self.up4 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.dec4 = AttentionConvBlock(1024 + self.ch4, 512)

        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec3 = AttentionConvBlock(512 + self.ch3, 256)

        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = AttentionConvBlock(256 + self.ch2, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = AttentionConvBlock(128 + self.ch1, 64)

        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
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

        b = self.bottleneck_attn(b)

        skip1 = e0
        skip2 = self.feb2(e2)
        skip3 = self.feb3(e3)
        skip4 = e4

        # Decoder 融合
        d4 = self.dec4(torch.cat([self.up4(b), skip4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))

        up_d2 = self.up1(d2)
        skip1_gated = self.ag1(g=up_d2, x=skip1)
        d1 = self.dec1(torch.cat([up_d2, skip1_gated], dim=1))

        # 主输出
        out = torch.sigmoid(self.final_conv(self.dec0(self.up0(d1))))

        if self.training:
            out1 = torch.sigmoid(self.out1(d1))
            out2 = torch.sigmoid(self.out2(d2))
            out3 = torch.sigmoid(self.out3(d3))
            return out, out1, out2, out3
        else:
            return out


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

    model = HANet_Final().to(device)

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