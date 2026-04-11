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
from tqdm import tqdm
import matplotlib.pyplot as plt  # 【新增】导入 matplotlib 用于绘图


# ==========================================
# 1. 评估指标与损失函数 (修改为标准的 BCE + Dice Loss)
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        # BCE Loss
        bce_loss = self.bce(pred, target)

        # Dice Loss
        smooth = 1e-5
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_score = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        dice_loss = 1.0 - dice_score

        return (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss.mean())


def calculate_metrics(pred, target, smooth=1e-5):
    """计算 ACC, PRECISION, RECALL, IOU, DICE"""
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
        "dice": dice.tolist(),
        "iou": iou.tolist(),
        "acc": acc.tolist(),
        "pre": pre.tolist(),
        "rec": rec.tolist()
    }


# ==========================================
# 2. HiFormer 模型基线 (完整核心结构)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class DoubleLevelFusion(nn.Module):
    def __init__(self, cnn_dim, vit_dim, out_dim):
        super().__init__()
        self.cnn_proj = nn.Sequential(nn.Conv2d(cnn_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim),
                                      nn.ReLU(True))
        self.vit_proj = nn.Sequential(nn.Conv2d(vit_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim),
                                      nn.ReLU(True))
        self.fusion_conv = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(out_dim), nn.ReLU(True))
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_dim, out_dim // 4, 1), nn.ReLU(True),
            nn.Conv2d(out_dim // 4, out_dim, 1), nn.Sigmoid()
        )

    def forward(self, cnn_feat, vit_feat):
        c_feat = self.cnn_proj(cnn_feat)
        v_feat = self.vit_proj(vit_feat)
        fused = self.fusion_conv(torch.cat([c_feat, v_feat], dim=1))
        return fused * self.channel_att(fused) + c_feat


class HiFormer_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. CNN 分支 (ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.cnn_layer1, self.cnn_layer2, self.cnn_layer3 = resnet.layer1, resnet.layer2, resnet.layer3

        # 2. Transformer 分支
        self.patch_embed = nn.Conv2d(3, 512, 16, 16)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 256, 512))  # 适配 224x224
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, activation='gelu', batch_first=True),
            num_layers=4
        )
        # 3. 交叉融合 DLF
        self.vit_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dlf_1 = DoubleLevelFusion(512, 512, 256)
        self.dlf_2 = DoubleLevelFusion(1024, 512, 512)

        # 4. Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(512 + 256, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(256 + 256, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(64, 32)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        e0 = self.stem(x)
        e1 = self.cnn_layer1(self.pool(e0))
        e2 = self.cnn_layer2(e1)
        e3 = self.cnn_layer3(e2)

        v_patch = self.patch_embed(x).flatten(2).transpose(1, 2)

        # 动态截取/适配位置编码以支持 224x224 (196 tokens)
        seq_len = v_patch.shape[1]
        pos_embed = self.pos_embed[:, :seq_len, :]

        v_trans = self.transformer(v_patch + pos_embed)
        v_feat = v_trans.transpose(1, 2).reshape(B, -1, H // 16, W // 16)

        feat_dlf2 = self.dlf_2(e3, v_feat)
        feat_dlf1 = self.dlf_1(e2, self.vit_up1(v_feat))

        d4 = self.dec4(torch.cat([self.up4(feat_dlf2), feat_dlf1], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e1], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e0], dim=1))
        out = torch.sigmoid(self.final_conv(self.dec1(self.up1(d2))))
        return out


# ==========================================
# 3. 数据集与加载器
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths, self.mask_paths, self.is_train = image_paths, mask_paths, is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = TF.resize(Image.open(self.image_paths[idx]).convert('RGB'), (224, 224))
        mask = TF.resize(Image.open(self.mask_paths[idx]).convert('L'), (224, 224))

        if self.is_train:
            if random.random() > 0.5: img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5: img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-15, 15)
            img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)

        return TF.to_tensor(img), TF.to_tensor(mask)


def get_dataset_paths(data_dir):
    img_p, mask_p = [], []
    for cat in ['benign', 'malignant']:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir): continue
        for f in os.listdir(cat_dir):
            if f.endswith('.png') and not f.endswith('_mask.png'):
                m_path = os.path.join(cat_dir, f.replace('.png', '_mask.png'))
                if os.path.exists(m_path):
                    img_p.append(os.path.join(cat_dir, f))
                    mask_p.append(m_path)
    return img_p, mask_p


# ==========================================
# 4. 主控台
# ==========================================
def main():
    # 🌟🌟🌟 控制面板 🌟🌟🌟
    MODE = "train"  # 改为 "train" 进行训练并生成图表

    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    save_path = "best_HiFormer.pth"
    plot_save_path = "training_curve.png"  # 【新增】图表保存路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | ⚙️ Mode: {MODE}")

    all_imgs, all_masks = get_dataset_paths(data_dir)
    if len(all_imgs) == 0:
        print("❌ 未找到数据！请检查路径。")
        return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total_size = len(all_imgs)
    t_size, v_size = int(0.8 * total_size), int(0.1 * total_size)

    model = HiFormer_Baseline().to(device)

    # ==========================
    # 模式 A：训练模式
    # ==========================
    if MODE == "train":
        train_loader = DataLoader(BUSIDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=10, shuffle=True)
        val_loader = DataLoader(BUSIDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False),
                                batch_size=10, shuffle=False)

        criterion = BCEDiceLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

        num_epochs = 45
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0

        # 【新增】用于记录每个 epoch 数据的列表
        history_train_loss = []
        history_val_dice = []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

            for images, masks in pbar_train:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), masks)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    metrics = calculate_metrics(model(images.to(device)), masks.to(device))
                    val_dices.extend(metrics["dice"])

            # 计算当前的 Loss 和 Dice
            avg_train_loss = train_loss_epoch / len(train_loader)
            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            # 【新增】将当前 epoch 的数据保存到记录列表中
            history_train_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best HiFormer Saved] Val Dice: {best_val_dice:.4f}")

        # 【新增】训练全部结束后，开始绘制训练图并保存
        print("\n📊 开始生成并保存训练曲线图...")
        plt.figure(figsize=(12, 5))

        # 绘制 Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), history_train_loss, marker='o', label='Train Loss', color='b')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # 绘制 Dice 曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), history_val_dice, marker='s', label='Validation Dice', color='orange')
        plt.title('Validation Dice Score per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.grid(True)
        plt.legend()

        # 调整布局并保存图片
        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=300)
        plt.close()
        print(f"✅ 训练曲线已成功保存至当前目录: {plot_save_path}")


    # ==========================
    # 模式 B：纯测试 / 评估模式
    # ==========================
    elif MODE == "test":
        print("\n" + "=" * 50)
        print("🚀 开始在测试集上评估 HiFormer 最终性能...")

        test_loader = DataLoader(BUSIDataset(all_imgs[t_size + v_size:], all_masks[t_size + v_size:], False),
                                 batch_size=10,
                                 shuffle=False)

        if not os.path.exists(save_path):
            print(f"❌ 找不到权重文件: {save_path}！请先运行 train 模式训练。")
            return

        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

        test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}

        print("🔥 正在进行 GPU 预热 (Warm-up)...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
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
        print("\n🏆 HiFormer 最终测试集成绩 🏆")
        print(f"🔹 Dice (F1): {np.mean(test_res['dice']):.4f}")
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