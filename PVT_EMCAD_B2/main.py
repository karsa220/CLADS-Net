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
import timm  # 💡 核心引入：timm 库


# ==========================================
# 1. EMCAD 解码器网络 (高效多尺度解码)
# ==========================================
class EMCAD_Module(nn.Module):
    """
    高效多尺度卷积注意力解码模块 (Efficient Multi-scale Convolutional Attention Decoding)
    使用交叉大核深度卷积捕获多尺度上下文，替代沉重的自注意力
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

        # 深度卷积并行分支 (模拟 1x3+3x1, 1x5+5x1, 1x7+7x1 大核空间注意力)
        self.dw3_h = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), groups=out_channels)
        self.dw3_v = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=out_channels)

        self.dw5_h = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), groups=out_channels)
        self.dw5_v = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), groups=out_channels)

        self.dw7_h = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3), groups=out_channels)
        self.dw7_v = nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0), groups=out_channels)

        self.attn_proj = nn.Conv2d(out_channels, out_channels, 1)
        self.proj_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv1(x)
        identity = x

        # 多尺度空间门控注意力提取
        attn3 = self.dw3_v(self.dw3_h(x))
        attn5 = self.dw5_v(self.dw5_h(x))
        attn7 = self.dw7_v(self.dw7_h(x))

        attn = attn3 + attn5 + attn7
        attn = self.attn_proj(attn)

        # 注意力加权
        x = identity * torch.sigmoid(attn)
        return self.proj_out(x)


# ==========================================
# 2. PVT-EMCAD 完整架构 (Timm 预训练骨干)
# ==========================================
class PVT_EMCAD(nn.Module):
    """
    结合 Timm 预训练 PVT-v2-B2 骨干与 EMCAD 解码器的顶级分割架构
    """

    def __init__(self, n_classes=1):
        super().__init__()

        # 🌟 核心替换：使用 timm 一键加载预训练的 PVT-v2-B2
        # features_only=True 允许我们自动获取 1/4, 1/8, 1/16, 1/32 四个尺度的特征图
        print("📥 正在从 timm 加载 PVT-v2-B2 ImageNet 预训练权重...")
        self.encoder = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)

        # PVT-v2-B2 的 4 个 Stage 输出通道数固定为 [64, 128, 320, 512]
        dims = [64, 128, 320, 512]
        decoder_dim = 64

        # EMCAD 解码模块 (自顶向下级联)
        self.emcad4 = EMCAD_Module(dims[3], decoder_dim)
        self.emcad3 = EMCAD_Module(dims[2] + decoder_dim, decoder_dim)
        self.emcad2 = EMCAD_Module(dims[1] + decoder_dim, decoder_dim)
        self.emcad1 = EMCAD_Module(dims[0] + decoder_dim, decoder_dim)

        # 最后的上采样与预测头
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 从 1/4 还原到原图
        self.out_conv = nn.Conv2d(decoder_dim, n_classes, kernel_size=1)

    def forward(self, x):
        # 1. 编码 (利用 timm 预训练模型提取多尺度特征)
        # 返回列表: [c1(1/4), c2(1/8), c3(1/16), c4(1/32)]
        features = self.encoder(x)
        c1, c2, c3, c4 = features

        # 2. 解码 (EMCAD 级联融合)
        d4 = self.emcad4(c4)

        d3 = self.up(d4)
        d3 = torch.cat([d3, c3], dim=1)
        d3 = self.emcad3(d3)

        d2 = self.up(d3)
        d2 = torch.cat([d2, c2], dim=1)
        d2 = self.emcad2(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, c1], dim=1)
        d1 = self.emcad1(d1)

        # 3. 输出
        out = self.final_up(d1)
        logits = self.out_conv(out)
        return torch.sigmoid(logits)


# ==========================================
# 3. 混合损失函数与指标计算 (保持原样)
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
# 4. 数据集与增强加载
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

        # PVT 骨干对分辨率较敏感，锁定为 256x256
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 💡 使用 ImageNet 预训练权重时，通常需要对图像进行标准化
        # 这里为了保持代码极简未引入 Normalize，如果你发现收敛依然慢，可以在此加入：
        # TF.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    # 初始化 带有 timm 权重的 PVT-EMCAD 模型
    # ==========================================
    model = PVT_EMCAD(n_classes=1).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧩 PVT-EMCAD-B2 (Timm) 模型参数量: {total_params / 1e6:.4f} M")

    criterion = HybridLoss()

    if MODE == "train":
        # 💡 使用了预训练权重后，可以采用不同层的学习率衰减 (Layer Decay)
        # 这里为了简便，统一使用略小的学习率微调骨干
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
        num_epochs = 35
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Training Progress (PVT-EMCAD-B2 Pretrained)')
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

                # Transformer 建议使用梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                print(f"🌟 [New Best PVT-EMCAD Saved] Val Dice: {best_val_dice:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        plt.ioff()
        plt.savefig('pvtemcad_pretrained_training_curve.png', dpi=150)
        print("📈 训练曲线已保存为 pvtemcad_pretrained_training_curve.png")

    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print("🚀 开始在完全未见的【测试集 (Test Set)】上评估 PVT-EMCAD 最终性能...")

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

            print("\n🏆 PVT-EMCAD-B2 (Timm Pretrained) 最终测试集成绩 🏆")
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