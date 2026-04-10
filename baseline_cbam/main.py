import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


# ==========================================
# 1. 核心创新模块：CBAM (通道与空间注意力)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 使用 1x1 卷积代替全连接，并加入 ReLU 增加非线性表达能力
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上做最大和平均池化，然后拼接
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class AttentionConvBlock(nn.Module):
    """结合了标准卷积与 CBAM 注意力的解码块"""

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
        x = x * self.ca(x)  # 通道注意力：挑出对肿瘤敏感的特征图
        x = x * self.sa(x)  # 空间注意力：压制背景散斑噪声
        return x


# ==========================================
# 2. 网络主体：DenseUNet + CBAM 解码器
# (注意：跳跃连接是纯天然的，没有任何额外模块)
# ==========================================
class DenseUNet_CBAM(nn.Module):
    def __init__(self):
        super(DenseUNet_CBAM, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.encoder = densenet.features
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 256, 512, 1024

        # 解码器使用 AttentionConvBlock
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

        # 纯粹的跳跃连接：直接把 e4, e3, e2, e0 拼接到解码器中
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))

        out = torch.sigmoid(self.final_conv(self.dec0(self.up0(d1))))
        return out


# ==========================================
# 3. 混合损失函数与指标计算
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        jaccard_loss = 1.0 - (intersection + smooth) / (union + smooth)
        return bce_loss + jaccard_loss


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
# 4. 数据集加载
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths, self.mask_paths = [], []
        categories = ['benign']
        for cat in categories:
            cat_dir = os.path.join(data_dir, cat)
            if not os.path.exists(cat_dir): continue
            for file_name in os.listdir(cat_dir):
                if file_name.endswith('.png') and not file_name.endswith('_mask.png'):
                    img_path = os.path.join(cat_dir, file_name)
                    mask_path = os.path.join(cat_dir, file_name.replace('.png', '_mask.png'))
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        resize = transforms.Resize((256, 256))
        return transforms.ToTensor()(resize(image)), transforms.ToTensor()(resize(mask))


# ==========================================
# 5. 主控台：实时绘图、训练、验证与测试
# ==========================================
def main():
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    dataset = BUSIDataset(data_dir)
    total_size = len(dataset)
    if total_size == 0:
        print("❌ 未找到数据！请检查路径。")
        return

    train_size = int(0.7 * total_size)
    test_size = int(0.2 * total_size)
    val_size = total_size - train_size - test_size

    # 固定种子 42，保证所有消融实验的数据完全一致
    train_sub, val_sub, test_sub = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )


    train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_sub, batch_size=8, shuffle=False)

    print(f"✅ 数据加载完毕 | Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # 实例化 Baseline + CBAM 模型
    model = DenseUNet_CBAM().to(device)
    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    num_epochs = 20
    best_val_dice = 0.0

    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax2.set_ylabel('Validation Dice', color='tab:blue')
    plt.title('Baseline + CBAM Training Progress')

    line_loss, = ax1.plot([], [], color='tab:red', marker='o', label='Train Loss')
    line_dice, = ax2.plot([], [], color='tab:blue', marker='s', label='Val Dice')
    ax1.legend([line_loss, line_dice], ['Train Loss', 'Val Dice'], loc='center right')

    history_loss, history_val_dice, epochs_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0
        train_steps = 0

        pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Train", leave=False)
        for images, masks in pbar_train:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            train_steps += 1
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss_epoch / train_steps

        model.eval()
        val_dices = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dices, _ = calculate_metrics(outputs, masks)
                val_dices.extend(dices)

        avg_val_dice = np.mean(val_dices)
        scheduler.step(avg_val_dice)

        print(f"📉 Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

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
            torch.save(model.state_dict(), "baseline_cbam_best_model.pth")
            print(f"🌟 [New Best Model Saved] Val Dice: {best_val_dice:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    plt.ioff()
    plt.savefig('baseline_cbam_curve.png', dpi=150)

    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的【测试集 (Test Set)】上评估性能...")

    model.load_state_dict(torch.load("baseline_cbam_best_model.pth"))
    model.eval()
    test_dices, test_ious = [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dices, ious = calculate_metrics(outputs, masks)
            test_dices.extend(dices)
            test_ious.extend(ious)

    print("\n🏆 Baseline + CBAM 最终测试集成绩 🏆")
    print(f"Test Dice: {np.mean(test_dices):.4f}")
    print(f"Test IoU:  {np.mean(test_ious):.4f}")
    print("=" * 50)

    plt.show()


if __name__ == "__main__":
    main()