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
import matplotlib.pyplot as plt


# ==========================================
# 1. 评估指标与损失函数 (保持不变)
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_score = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
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
# 2. HiFormer 模型基线 (保持不变)
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
        self.channel_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_dim, out_dim // 4, 1), nn.ReLU(True),
                                         nn.Conv2d(out_dim // 4, out_dim, 1), nn.Sigmoid())

    def forward(self, cnn_feat, vit_feat):
        c_feat = self.cnn_proj(cnn_feat)
        v_feat = self.vit_proj(vit_feat)
        fused = self.fusion_conv(torch.cat([c_feat, v_feat], dim=1))
        return fused * self.channel_att(fused) + c_feat


class HiFormer_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.cnn_layer1, self.cnn_layer2, self.cnn_layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        self.patch_embed = nn.Conv2d(3, 512, 16, 16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 512))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, activation='gelu', batch_first=True),
            num_layers=4)
        self.vit_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dlf_1 = DoubleLevelFusion(512, 512, 256)
        self.dlf_2 = DoubleLevelFusion(1024, 512, 512)
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
        v_trans = self.transformer(v_patch + self.pos_embed[:, :v_patch.shape[1], :])
        v_feat = v_trans.transpose(1, 2).reshape(B, -1, H // 16, W // 16)
        feat_dlf2 = self.dlf_2(e3, v_feat)
        feat_dlf1 = self.dlf_1(e2, self.vit_up1(v_feat))
        d4 = self.dec4(torch.cat([self.up4(feat_dlf2), feat_dlf1], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e1], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e0], dim=1))
        out = torch.sigmoid(self.final_conv(self.dec1(self.up1(d2))))
        return out


# ==========================================
# 3. UDIAT 数据集加载逻辑 (重点修改部分)
# ==========================================
class UDIATDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和掩码
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 调整大小
        img = TF.resize(img, (224, 224))
        mask = TF.resize(mask, (224, 224))

        if self.is_train:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-15, 15)
            img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)

        return TF.to_tensor(img), TF.to_tensor(mask)


def get_udiat_paths(data_dir):
    """
    针对 UDIAT 数据集结构：
    data_dir/
      ├── original/ (000001.PNG, ...)
      └── GT/       (000001.PNG, ...)
    """
    img_dir = os.path.join(data_dir, 'original')
    gt_dir = os.path.join(data_dir, 'GT')

    img_p, mask_p = [], []

    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"❌ 路径不存在: 请确保存在 {img_dir} 和 {gt_dir}")
        return [], []

    # 遍历 original 文件夹
    filenames = sorted(os.listdir(img_dir))
    for f in filenames:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(gt_dir, f)  # UDIAT 通常原图和掩码文件名完全一致

            if os.path.exists(mask_path):
                img_p.append(img_path)
                mask_p.append(mask_path)
            else:
                # 兼容部分数据集大小写不一致的情况
                alt_mask_path = os.path.join(gt_dir,
                                             f.replace('.PNG', '.png') if '.PNG' in f else f.replace('.png', '.PNG'))
                if os.path.exists(alt_mask_path):
                    img_p.append(img_path)
                    mask_p.append(alt_mask_path)

    print(f"✅ 成功找到 {len(img_p)} 对 UDIAT 图像和掩码。")
    return img_p, mask_p


# ==========================================
# 4. 主控台
# ==========================================
def main():
    # 🌟🌟🌟 配置面板 🌟🌟🌟
    MODE = "test"
    data_dir = r"D:\PycharmProjects\data\UDIAT_Dataset_B"  # 修改为你的 UDIAT 数据集根目录
    save_path = "best_HiFormer_UDIAT.pth"
    plot_save_path = "training_curve_udiat.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | ⚙️ Mode: {MODE}")

    # 获取 UDIAT 路径
    all_imgs, all_masks = get_udiat_paths(data_dir)
    if len(all_imgs) == 0:
        return

    # 打乱并划分
    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total_size = len(all_imgs)
    # 划分比例: 80% 训练, 10% 验证, 10% 测试
    t_size = int(0.8 * total_size)
    v_size = int(0.1 * total_size)

    model = HiFormer_Baseline().to(device)

    if MODE == "train":
        train_loader = DataLoader(UDIATDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=8, shuffle=True)
        val_loader = DataLoader(
            UDIATDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False), batch_size=8,
            shuffle=False)

        criterion = BCEDiceLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        num_epochs = 40
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0
        history_train_loss, history_val_dice = [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = criterion(out, masks)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    metrics = calculate_metrics(model(images.to(device)), masks.to(device))
                    val_dices.extend(metrics["dice"])

            avg_train_loss = train_loss_epoch / len(train_loader)
            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            history_train_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Saved New Best Model!")

        # 绘图
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1);
        plt.plot(history_train_loss);
        plt.title("Loss")
        plt.subplot(1, 2, 2);
        plt.plot(history_val_dice);
        plt.title("Val Dice")
        plt.savefig(plot_save_path);
        plt.close()
        print(f"✅ 训练完成，曲线已保存。")

    # ==========================
    # 纯测试 / 评估模式
    # ==========================
    test_loader = DataLoader(UDIATDataset(all_imgs[t_size + v_size:], all_masks[t_size + v_size:], False),
                             batch_size=1, shuffle=False)
    if not os.path.exists(save_path):
        print(f"❌ 找不到模型权重文件: {save_path}")
        return

    model.load_state_dict(torch.load(save_path))
    model.eval()

    # 👉 修改点 1：在这里把缺少的 3 个指标（acc, pre, rec）加上
    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            outputs = model(images.to(device))
            m = calculate_metrics(outputs, masks.to(device))

            # 👉 修改点 2：把所有指标都存进字典里
            for key in test_res.keys():
                test_res[key].extend(m[key])

    # 👉 修改点 3：把所有指标都打印出来
    print(f"\n🏆 UDIAT 测试结果:")
    print(f"🔹 Dice (F1): {np.mean(test_res['dice']):.4f}")
    print(f"🔹 IoU      : {np.mean(test_res['iou']):.4f}")
    print(f"🔹 ACC      : {np.mean(test_res['acc']):.4f}")
    print(f"🔹 Precision: {np.mean(test_res['pre']):.4f}")
    print(f"🔹 Recall   : {np.mean(test_res['rec']):.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()