import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import time
import argparse

# ==========================================
# 1. 导入官方的 config 和 Swin-Unet 模型
# 请确保 config.py, networks 文件夹和 configs 文件夹在当前目录下
# ==========================================
from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg


# ==========================================
# 2. 官方 Swin-Unet 的包装器
# ==========================================
class OfficialSwinUNetWrapper(nn.Module):
    def __init__(self, img_size=224, num_classes=1, cfg_path='configs/swin_tiny_patch4_window7_224_lite.yaml'):
        super(OfficialSwinUNetWrapper, self).__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=cfg_path, help='path to config file')
        parser.add_argument("--opts", help="Modify config options", default=None, nargs='+')
        parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
        parser.add_argument('--zip', action='store_true')
        parser.add_argument('--cache-mode', type=str, default='part')
        parser.add_argument('--resume', help='resume from checkpoint')
        parser.add_argument('--accumulation-steps', type=int)
        parser.add_argument('--use-checkpoint', action='store_true')
        parser.add_argument('--amp-opt-level', type=str, default='O1')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--throughput', action='store_true')
        args = parser.parse_args(args=[])

        config = get_config(args)
        self.model = ViT_seg(config, img_size=img_size, num_classes=num_classes)

        if os.path.exists(config.MODEL.PRETRAIN_CKPT):
            self.model.load_from(config)
        else:
            print(f"⚠️ 未找到预训练权重 {config.MODEL.PRETRAIN_CKPT}，将随机初始化从头开始训练。")

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)


# ==========================================
# 3. 混合损失函数与指标计算 (替换为官方策略)
# ==========================================
class OfficialDiceLoss(nn.Module):
    """提取自官方 utils.py 的 Dice 计算逻辑"""

    def __init__(self):
        super(OfficialDiceLoss, self).__init__()

    def forward(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1.0 - loss


class OfficialHybridLoss(nn.Module):
    """匹配官方 trainer.py 的 loss 组合: 0.4 * CE(此处对应BCE) + 0.6 * Dice"""

    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(OfficialHybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = OfficialDiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss)


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

        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224))

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 官方默认使用 -20 ~ 20 的随机旋转
            angle = random.uniform(-20, 20)
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
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

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
    MODE = "train"

    model = OfficialSwinUNetWrapper(img_size=224, num_classes=1).to(device)

    # 使用修改为官方的组合损失： 0.4 BCE + 0.6 Dice
    criterion = OfficialHybridLoss()
    save_path = "best_busi.pth"

    if MODE == "train":
        # 官方使用 SGD 而不是 Adam，使用 momentum 和 weight_decay
        # 根据 batch size，通常基础学习率在 0.01 到 0.05 之间，这里使用 default=0.01
        base_lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
        num_epochs = 35

        # 官方使用的按 iteration 更新的 poly decay，无需 pytorch 自带的 Scheduler
        iter_num = 0
        max_iterations = num_epochs * len(train_loader)

        best_val_dice = 0.0

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Official Swin-Unet Training Progress')
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

                # 按照官方源码 trainer.py 中的 Poly Decay 更新学习率
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num += 1

                train_loss_epoch += loss.item()
                pbar_train.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{lr_:.6f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_dices.extend(calculate_metrics(outputs, masks)["dice"])

            avg_val_dice = np.mean(val_dices)

            print(
                f"📉 Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | End LR: {lr_:.6f}")

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

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Baseline Saved] Val Dice: {best_val_dice:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        plt.ioff()
        plt.savefig('baseline_training_curve.png', dpi=150)
        print("📈 基线训练曲线已保存为 baseline_training_curve.png")

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
            print("\n🏆 swinunet (BUSI dataset)  最终测试集成绩 🏆")
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