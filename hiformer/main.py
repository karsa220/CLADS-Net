import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


from models.HiFormer import HiFormer
from configs.HiFormer_configs import get_hiformer_b_configs, get_hiformer_s_configs, get_hiformer_l_configs


# ==========================================
# 1. 评估指标与损失函数
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(HybridLoss, self).__init__()
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
# 2. 数据集与加载器
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        # 【修复】这里补上了等号右边的 is_train
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
# 3. 主控台
# ==========================================
def main():
    # 🌟🌟🌟 控制面板 🌟🌟🌟
    MODE = "train"

    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    save_path = "best_busi.pth"
    plot_save_path = "training_curve.png"

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


    config = get_hiformer_b_configs()  # 获取 B 版本的配置 (您也可以改为 _s_configs 或 _l_configs)

    # 设定 n_classes=1，因为 BUSI 数据集的 mask 是单通道目标
    model = HiFormer(config=config, img_size=224, in_chans=3, n_classes=1).to(device)

    # ==========================
    # 模式 A：训练模式
    # ==========================
    if MODE == "train":
        train_loader = DataLoader(BUSIDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=10, shuffle=True)
        val_loader = DataLoader(BUSIDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False),
                                batch_size=10, shuffle=False)

        criterion = HybridLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        num_epochs = 35
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0
        history_train_loss = []
        history_val_dice = []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

            for images, masks in pbar_train:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                # 【修改】获取官方模型输出并经过 Sigmoid 激活以适配 BCELoss
                outputs = model(images)
                outputs = torch.sigmoid(outputs)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    outputs = model(images.to(device))
                    outputs = torch.sigmoid(outputs)  # 【修改】验证时同样需要 Sigmoid
                    metrics = calculate_metrics(outputs, masks.to(device))
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
                print(f"🌟 [New Best HiFormer Saved] Val Dice: {best_val_dice:.4f}")

        print("\n📊 开始生成并保存训练曲线图...")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), history_train_loss, marker='o', label='Train Loss', color='b')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), history_val_dice, marker='s', label='Validation Dice', color='orange')
        plt.title('Validation Dice Score per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=300)
        plt.close()
        print(f"✅ 训练曲线已成功保存至当前目录: {plot_save_path}")

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
            outputs = torch.sigmoid(outputs)  # 【修改】测试时同样需要 Sigmoid 取出概率值

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