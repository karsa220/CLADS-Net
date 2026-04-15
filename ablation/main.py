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
from tqdm import tqdm


# ==========================================
# 1. 基础模块 (保持不变)
# ==========================================
class CrossScaleMLPAttention(nn.Module):
    def __init__(self, channels, num_scales=5, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        mid_channels = max(1, (channels * num_scales) // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels * num_scales, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, channels * num_scales, bias=False)
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        stacked_f = torch.stack(features, dim=1)
        squeezed = stacked_f.mean(dim=(3, 4))
        squeezed = squeezed.view(B, -1)
        attn_weights = self.mlp(squeezed)
        attn_weights = attn_weights.view(B, self.num_scales, C)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_weights = attn_weights.view(B, self.num_scales, C, 1, 1)
        fused_out = (stacked_f * attn_weights).sum(dim=1)
        return fused_out


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


class LeakyRCAB(nn.Module):
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


# ==========================================
# 2. 支持消融控制的模型 (核心修改区)
# ==========================================
class CLADS_Net_Ablation(nn.Module):
    def __init__(self, embed_dim=256, use_rcab=True, use_csma=True, use_deep_sup=True):
        super().__init__()
        self.use_rcab = use_rcab
        self.use_csma = use_csma
        self.use_deep_sup = use_deep_sup

        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features

        self.ch0, self.ch2, self.ch3, self.ch4, self.ch_b = 64, 256, 512, 1024, 1024

        # 消融 RCAB
        if self.use_rcab:
            self.rcab2 = LeakyRCAB(self.ch2)
            self.rcab3 = LeakyRCAB(self.ch3)
            self.rcab4 = LeakyRCAB(self.ch4)

        self.linear0 = MLP(self.ch0, embed_dim)
        self.linear2 = MLP(self.ch2, embed_dim)
        self.linear3 = MLP(self.ch3, embed_dim)
        self.linear4 = MLP(self.ch4, embed_dim)
        self.linear_b = MLP(self.ch_b, embed_dim)

        # 消融 CSMA
        if self.use_csma:
            self.csma_fuse = CrossScaleMLPAttention(channels=embed_dim, num_scales=5, reduction=4)
        else:
            # 如果不用 CSMA，则回退到基础的 Cat + Conv 融合方式
            self.simple_fuse = nn.Sequential(
                nn.Conv2d(embed_dim * 5, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )

        self.final_conv = nn.Conv2d(embed_dim, 1, 1)

        # 消融 Deep Supervision
        if self.use_deep_sup:
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

        # 根据配置决定是否走 RCAB
        s2 = self.rcab2(e2) if self.use_rcab else e2
        s3 = self.rcab3(e3) if self.use_rcab else e3
        s4 = self.rcab4(e4) if self.use_rcab else e4

        target_size = e0.shape[2:]

        l0 = self.linear0(e0)
        l2 = F.interpolate(self.linear2(s2), target_size, mode='bilinear', align_corners=False)
        l3 = F.interpolate(self.linear3(s3), target_size, mode='bilinear', align_corners=False)
        l4 = F.interpolate(self.linear4(s4), target_size, mode='bilinear', align_corners=False)
        lb = F.interpolate(self.linear_b(b), target_size, mode='bilinear', align_corners=False)

        # 根据配置决定融合方式
        if self.use_csma:
            fused = self.csma_fuse(l0, l2, l3, l4, lb)
        else:
            stacked = torch.cat([l0, l2, l3, l4, lb], dim=1)
            fused = self.simple_fuse(stacked)

        out = torch.sigmoid(
            F.interpolate(self.final_conv(fused), size=x.shape[2:], mode='bilinear', align_corners=False))

        # 根据配置决定是否返回辅助损失输出
        if self.training and self.use_deep_sup:
            out0 = torch.sigmoid(F.interpolate(self.aux0(l0), size=x.shape[2:], mode='bilinear', align_corners=False))
            out2 = torch.sigmoid(F.interpolate(self.aux2(l2), size=x.shape[2:], mode='bilinear', align_corners=False))
            out3 = torch.sigmoid(F.interpolate(self.aux3(lb), size=x.shape[2:], mode='bilinear', align_corners=False))
            out4 = torch.sigmoid(F.interpolate(self.aux4(l4), size=x.shape[2:], mode='bilinear', align_corners=False))
            return out, out0, out2, out3, out4

        return out


# ==========================================
# 3. 损失函数与指标
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super().__init__()
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
# 4. 数据集
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
# 5. 消融实验主流程
# ==========================================
def main():
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"  # 你的数据集路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 数据划分
    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ 未找到数据集！")
        return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    train_loader = DataLoader(BUSIDataset(all_imgs[:train_size], all_masks[:train_size], True), batch_size=8,
                              shuffle=True)
    val_loader = DataLoader(
        BUSIDataset(all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size], False),
        batch_size=8)
    test_loader = DataLoader(BUSIDataset(all_imgs[train_size + val_size:], all_masks[train_size + val_size:], False),
                             batch_size=8, shuffle=False)

    # 🎯 定义消融实验配置字典
    ablation_configs = {
        # 2. 单模块有效性验证 (加法逻辑：开一关二)
        "2. + RCAB Only": {"use_rcab": True, "use_csma": False, "use_deep_sup": False},
        "3. + CSMA Only": {"use_rcab": False, "use_csma": True, "use_deep_sup": False},
        "4. + Deep Sup. Only": {"use_rcab": False, "use_csma": False, "use_deep_sup": True},

      #  "w/o CSMA (No Cross-Scale Attn)": {"use_rcab": True, "use_csma": False, "use_deep_sup": True},
      #  "w/o RCAB (No Channel Attn)": {"use_rcab": False, "use_csma": True, "use_deep_sup": True},
      #  "w/o DeepSup (No Aux Loss)": {"use_rcab": True, "use_csma": True, "use_deep_sup": False},
       # "Vanilla (No Custom Modules)": {"use_rcab": False, "use_csma": False, "use_deep_sup": False},
    }

    results = {}
    num_epochs = 15
    criterion = HybridLoss()

    for config_name, kwargs in ablation_configs.items():
        print("\n" + "=" * 60)
        print(f"🔬 开始消融实验: {config_name}")
        print(f"⚙️ 参数: {kwargs}")
        print("=" * 60)

        model = CLADS_Net_Ablation(embed_dim=256, **kwargs).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0
        best_model_weights = None

        # --- 训练阶段 ---
        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}")

            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                if kwargs["use_deep_sup"]:
                    out, out0, out2, out3, out4 = model(images)
                    loss_main = criterion(out, masks)
                    loss_aux = (0.4 * criterion(out0, masks) + 0.4 * criterion(out2, masks) +
                                0.4 * criterion(out3, masks) + 0.4 * criterion(out4, masks))
                    loss = loss_main + 0.4 * loss_aux
                else:
                    out = model(images)
                    loss = criterion(out, masks)

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            # --- 验证阶段 ---
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    # 如果测试也返回了多个头（理论上 eval 模式只返回1个，兼容处理）
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    val_dices.extend(calculate_metrics(outputs, masks)["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                best_model_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"✅ {config_name} 训练完成！最佳验证集 Dice: {best_val_dice:.4f}")

        # --- 测试阶段 (加载最佳权重并在 Test Set 评估) ---
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
        model.eval()
        test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple): outputs = outputs[0]

                metrics = calculate_metrics(outputs, masks)
                for k in test_res.keys():
                    test_res[k].extend(metrics[k])

        # 记录结果
        results[config_name] = {
            "Dice": np.mean(test_res["dice"]),
            "IoU": np.mean(test_res["iou"]),
            "ACC": np.mean(test_res["acc"]),
            "Pre": np.mean(test_res["pre"]),
            "Rec": np.mean(test_res["rec"]),
        }

    # ==========================================
    # 6. 打印最终的消融实验对比表格
    # ==========================================
    print("\n\n" + "🏆" * 5 + " 消融实验最终结果对比表 " + "🏆" * 5)
    print(f"| {'Model Configuration':<30} | {'Dice':<6} | {'IoU':<6} | {'ACC':<6} | {'Precision':<9} | {'Recall':<6} |")
    print(f"|{'-' * 32}|{'-' * 8}|{'-' * 8}|{'-' * 8}|{'-' * 11}|{'-' * 8}|")
    for name, metrics in results.items():
        print(
            f"| {name:<30} | {metrics['Dice']:.4f} | {metrics['IoU']:.4f} | {metrics['ACC']:.4f} | {metrics['Pre']:.4f}    | {metrics['Rec']:.4f} |")
    print("=" * 79)


if __name__ == "__main__":
    main()