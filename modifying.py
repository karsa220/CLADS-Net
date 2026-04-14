
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


class MLP(nn.Module):
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
    def __init__(self, in_channels):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(in_channels // 4, 1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(mid, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        feat = self.local_conv(x)
        att = self.channel_att(self.gap(feat))
        return feat * att + identity


class HANet_MLP_Final(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features

        self.ch0, self.ch2, self.ch3 = 64, 256, 512
        self.ch4, self.ch_b = 1024, 1024

        self.feb2 = Basic_FEB_Leaky(self.ch2)
        self.feb3 = Basic_FEB_Leaky(self.ch3)
        self.feb4 = Basic_FEB_Leaky(self.ch4)

        self.linear0 = MLP(self.ch0, embed_dim)
        self.linear2 = MLP(self.ch2, embed_dim)
        self.linear3 = MLP(self.ch3, embed_dim)
        self.linear4 = MLP(self.ch4, embed_dim)
        self.linear_b = MLP(self.ch_b, embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 5, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.final_conv = nn.Conv2d(embed_dim, 1, 1)

        self.aux0 = nn.Conv2d(embed_dim, 1, 1)
        self.aux2 = nn.Conv2d(embed_dim, 1, 1)
        self.aux3 = nn.Conv2d(embed_dim, 1, 1)
        self.aux4 = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, x):
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x)))
        e2 = self.encoder.denseblock1(self.encoder.pool0(e0))
        e3 = self.encoder.denseblock2(self.encoder.transition1(e2))
        e4 = self.encoder.denseblock3(self.encoder.transition2(e3))
        b = self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e4)))

        s2 = self.feb2(e2)
        s3 = self.feb3(e3)
        s4 = self.feb4(e4)

        target_size = e0.shape[2:]

        l0 = self.linear0(e0)
        l2 = F.interpolate(self.linear2(s2), target_size, mode='bilinear', align_corners=False)
        l3 = F.interpolate(self.linear3(s3), target_size, mode='bilinear', align_corners=False)
        l4 = F.interpolate(self.linear4(s4), target_size, mode='bilinear', align_corners=False)
        lb = F.interpolate(self.linear_b(b), target_size, mode='bilinear', align_corners=False)

        fused = self.linear_fuse(torch.cat([l0, l2, l3, l4, lb], dim=1))

        out = torch.sigmoid(F.interpolate(self.final_conv(fused), size=x.shape[2:], mode='bilinear', align_corners=False))

        if self.training:
            out0 = torch.sigmoid(F.interpolate(self.aux0(l0), size=x.shape[2:], mode='bilinear', align_corners=False))
            out2 = torch.sigmoid(F.interpolate(self.aux2(l2), size=x.shape[2:], mode='bilinear', align_corners=False))
            out3 = torch.sigmoid(F.interpolate(self.aux3(lb), size=x.shape[2:], mode='bilinear', align_corners=False))
            out4 = torch.sigmoid(F.interpolate(self.aux4(l4), size=x.shape[2:], mode='bilinear', align_corners=False))
            return out, out0, out2, out3, out4
        return out
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
    return {"dice": dice.tolist(), "iou": iou.tolist(), "acc": acc.tolist(), "pre": pre.tolist(), "rec": rec.tolist()}


# ==========================================
# 4. 数据加载与主流程 (已适配新模型)
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
    MODE = "train"
    save_path = "best_model_mlp_decoder.pth"
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"  # 请根据实际修改

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 📐 Architecture: HANet-MLP")

    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0: return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # 实例化新模型
    model = HANet_MLP_Final(embed_dim=256).to(device)

    if MODE == "train":
        train_loader = DataLoader(BUSIDataset(all_imgs[:train_size], all_masks[:train_size], True), batch_size=8,
                                  shuffle=True)
        val_loader = DataLoader(
            BUSIDataset(all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size], False),
            batch_size=8)

        criterion = HybridLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # MLP 适合用 AdamW
        num_epochs = 15
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0
        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                # 新的输出结构
                out, out0, out2, out3,out4 = model(images)
                loss_main = criterion(out, masks)
                loss_aux = (
                        0.4 * criterion(out0, masks) +
                        0.4 * criterion(out2, masks) +
                        0.4 * criterion(out3, masks) +
                        0.4 * criterion(out4, masks)
                )

                loss = loss_main + 0.4 * loss_aux

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

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