import os
import random
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. 导入 MISSFormer 并进行 Sigmoid 包装
# ==========================================
from networks.MISSFormer import MISSFormer


class MISSFormerWrapper(nn.Module):
    def __init__(self, n_classes=1):
        super(MISSFormerWrapper, self).__init__()
        self.model = MISSFormer(num_classes=n_classes)

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)


# ==========================================
# 2. 修正后的二分类 Loss 与指标计算
# ==========================================
class AuthorStyleLoss(nn.Module):
    def __init__(self):
        super(AuthorStyleLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        smooth = 1e-5
        intersect = torch.sum(pred * target)

        # 🚀 必须修正为标准的二分类 Dice 分母，防止平方导致的梯度消失
        dice = (2 * intersect + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
        loss_dice = 1 - dice

        return 0.4 * loss_bce + 0.6 * loss_dice


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
# 3. 数据集加载
# ==========================================
class UDIATDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths, self.mask_paths, self.is_train = image_paths, mask_paths, is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 强制 224x224
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224))

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                translate = [int(random.uniform(-10, 10)), int(random.uniform(-10, 10))]
                image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0)
                mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # MISSFormer 标准归一化
        image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return image, mask


def get_udiat_paths(data_dir):
    img_dir, gt_dir = os.path.join(data_dir, 'original'), os.path.join(data_dir, 'GT')
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"❌ 路径错误: 找不到 original 或 GT 文件夹于 {data_dir}")
        return [], []

    img_p, mask_p = [], []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path, mask_path = os.path.join(img_dir, f), os.path.join(gt_dir, f)
            if os.path.exists(mask_path):
                img_p.append(img_path)
                mask_p.append(mask_path)
            else:
                alt_f = f.replace('.PNG', '.png') if '.PNG' in f else f.replace('.png', '.PNG')
                alt_mask_path = os.path.join(gt_dir, alt_f)
                if os.path.exists(alt_mask_path):
                    img_p.append(img_path)
                    mask_p.append(alt_mask_path)

    print(f"✅ 成功找到 {len(img_p)} 对 UDIAT 图像和掩码。")
    return img_p, mask_p


# ==========================================
# 4. 主控台
# ==========================================
def main():
    # 🌟🌟🌟 控制面板 🌟🌟🌟
    MODE = "train"  # 先设为 train 跑一次
    data_dir = r"D:\PycharmProjects\data\UDIAT_Dataset_B"

    pretrained_busi_path = "best_busi.pth"
    mit_imagenet_path = "mit_b1.pth"  # ⚠️ 确保此文件在你项目目录下
    save_path = "best_UDIAT_finetuned.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | ⚙️ Mode: {MODE}")

    all_imgs, all_masks = get_udiat_paths(data_dir)
    if not all_imgs: return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total = len(all_imgs)
    t_size, v_size = int(0.8 * total), int(0.1 * total)

    model = MISSFormerWrapper(n_classes=1).to(device)

    if MODE == "train":
        # 🚀 权重加载逻辑：优先加载 BUSI，没有则回退加载 ImageNet (mit_b1.pth)
        if os.path.exists(pretrained_busi_path):
            print(f"🔄 加载 BUSI 预训练权重: {pretrained_busi_path}")
            model.load_state_dict(torch.load(pretrained_busi_path, map_location=device))
        elif os.path.exists(mit_imagenet_path):
            print(f"🔄 未找到BUSI权重，加载官方 MiT-B1 ImageNet 预训练权重: {mit_imagenet_path}")
            pretrained_dict = torch.load(mit_imagenet_path, map_location='cpu')
            model_dict = model.model.backbone.state_dict()
            new_state_dict = {}
            for k, v in pretrained_dict.items():
                new_k = k.replace('backbone.', '').replace('head.', '')
                if new_k in model_dict and v.size() == model_dict[new_k].size():
                    new_state_dict[new_k] = v
            model.model.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"✅ 成功加载了 {len(new_state_dict)} 个 MiT-B1 张量！")
        else:
            print("⚠️ 警告: 未找到任何预训练权重，模型从头训练，跑分极有可能崩溃！")

        train_loader = DataLoader(UDIATDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=8, shuffle=True)
        val_loader = DataLoader(
            UDIATDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False), batch_size=8,
            shuffle=False)

        criterion = AuthorStyleLoss()

        # 🚀 必须使用差分学习率！Backbone 小学习率，Decoder 大学习率
        backbone_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4},  # Backbone 微调
            {'params': decoder_params, 'lr': 1e-3}  # Decoder 快速学习
        ], weight_decay=1e-4)

        num_epochs = 50
        max_iterations = num_epochs * len(train_loader)

        # 🚀 余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-6)

        best_val_dice = 0.0
        history_loss, history_val_dice = [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()

                # 🚀 梯度裁剪，防止 Transformer 在 1e-3 学习率下梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()  # 每个 iter 步进一次

                # 获取 Decoder 的学习率用于显示
                current_lr = optimizer.param_groups[1]['lr']
                train_loss_epoch += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR(Dec)': f"{current_lr:.6f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # 验证评估
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    out = model(images.to(device))
                    val_dices.extend(calculate_metrics(out, masks.to(device))["dice"])

            avg_val_dice = np.mean(val_dices)
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(
                f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Dec LR: {current_lr:.6f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Saved] Val Dice: {best_val_dice:.4f} -> Saved to {save_path}")

            torch.cuda.empty_cache()
            gc.collect()

        # 绘制训练图表
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_loss, marker='o')
        plt.title("Train Loss")
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(history_val_dice, marker='s', color='orange')
        plt.title("Val Dice")
        plt.grid()
        plt.tight_layout()
        plt.savefig('training_curve_MISSFormer_UDIAT_Finetuned.png', dpi=150)
        print("✅ 训练完成，图表已保存。")

    # ==========================================
    # 5. 测试模式 (Test)
    # ==========================================
    if MODE in ["train", "test"]:
        test_loader = DataLoader(UDIATDataset(all_imgs[t_size + v_size:], all_masks[t_size + v_size:], False),
                                 batch_size=1, shuffle=False)
        print("\n" + "=" * 50)
        print("🚀 开始在完全未见的【测试集】上评估 MISSFormer...")

        if not os.path.exists(save_path):
            print(f"❌ 找不到权重 {save_path}！请先运行 train。")
        else:
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
            print("🔥 GPU 预热 (Warm-up)...")
            with torch.no_grad():
                for _ in range(5): model(torch.randn(1, 3, 224, 224).to(device))
            if torch.cuda.is_available(): torch.cuda.synchronize()

            total_infer_time, total_samples = 0.0, 0

            with torch.no_grad():
                for images, masks in tqdm(test_loader, desc="Testing"):
                    images, masks = images.to(device), masks.to(device)

                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    start_time = time.time()
                    outputs = model(images)
                    if torch.cuda.is_available(): torch.cuda.synchronize()

                    total_infer_time += (time.time() - start_time)
                    total_samples += images.size(0)

                    metrics = calculate_metrics(outputs, masks)
                    for k in test_res.keys(): test_res[k].extend(metrics[k])

            avg_time_per_image = (total_infer_time / total_samples) * 1000
            fps = 1.0 / (total_infer_time / total_samples)

            print("\n🏆 MISSFormer (UDIAT Finetuned) 测试集最终成绩 🏆")
            print(f"🔹 Dice     : {np.mean(test_res['dice']):.4f}")
            print(f"🔹 IoU      : {np.mean(test_res['iou']):.4f}")
            print(f"🔹 ACC      : {np.mean(test_res['acc']):.4f}")
            print(f"🔹 Precision: {np.mean(test_res['pre']):.4f}")
            print(f"🔹 Recall   : {np.mean(test_res['rec']):.4f}")
            print("-" * 50)
            print(f"⏱️ 平均耗时 : {avg_time_per_image:.2f} ms/img | FPS: {fps:.2f}")
            print("=" * 50)


if __name__ == "__main__":
    main()