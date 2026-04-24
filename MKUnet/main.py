import os
import random
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心导入：直接使用官方的模块
# ==========================================
# 导入官方 MK-UNet 模型
from mkunet_network import MK_UNet
# 导入官方针对医学分割设计的加权结构损失函数
from train_polyp import structure_loss


# ==========================================
# 2. 指标计算
# ==========================================
def calculate_metrics(pred_probs, target, smooth=1e-5):
    # 注意：这里的 pred_probs 必须是经过 sigmoid 激活后的概率图 (0~1)
    pred_bin = (pred_probs > 0.5).float()
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
# 3. 数据集与增强加载 (使用你的原有逻辑)
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

        # 官方代码默认使用 352x352 或者是 256x256，这里保留你的 256
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
# 4. 主控台
# ==========================================
def main():
    MODE = "train"  # 训练完后可以改成 "test" 仅测试
    save_path = "best_busi.pth"

    # 请确保路径正确
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

    # 官方代码通常使用的 batch_size 偏小 (如 8)，你可以根据显存自行调整
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ 数据加载完毕 | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ==========================================
    # 实例化官方模型
    # ==========================================
    model = MK_UNet(num_classes=1, in_channels=3).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧩 官方 MK-UNet 模型参数量: {total_params / 1e6:.4f} M")

    if MODE == "train":
        # 官方代码中默认的学习率是 0.0005，使用 AdamW
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        num_epochs = 35
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Training Progress (Official MK-UNet)')
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

                # 官方模型返回的是一个列表，我们取主要输出
                logits = outputs[0] if isinstance(outputs, list) else outputs

                # 官方 structure_loss 内部包含了 BCEWithLogits，所以直接传入 logits
                loss = structure_loss(logits, masks)
                loss.backward()

                # 官方代码中使用了梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                train_loss_epoch += loss.item()
                pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # --- Validation 阶段 ---
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    logits = outputs[0] if isinstance(outputs, list) else outputs

                    # 在计算 Dice 之前，需要手动将 logits 转换为概率 (Sigmoid)
                    probs = torch.sigmoid(logits)
                    val_dices.extend(calculate_metrics(probs, masks)["dice"])

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
                print(f"🌟 [New Best Official MK-UNet Saved] Val Dice: {best_val_dice:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        plt.ioff()
        plt.savefig('official_mkunet_training_curve.png', dpi=150)
        print("📈 训练曲线已保存为 official_mkunet_training_curve.png")

    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print("🚀 开始在完全未见的【测试集 (Test Set)】上评估 Official MK-UNet ...")

        if not os.path.exists(save_path):
            print(f"❌ 找不到权重文件: {save_path}！请先运行 train 模式训练。")
        else:
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}

            # GPU 预热以准确测速
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
                    logits = outputs[0] if isinstance(outputs, list) else outputs
                    probs = torch.sigmoid(logits)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()

                    total_infer_time += (end_time - start_time)
                    total_samples += images.size(0)

                    metrics = calculate_metrics(probs, masks)
                    for k in test_res.keys():
                        test_res[k].extend(metrics[k])

            avg_time_per_image = (total_infer_time / total_samples) * 1000
            fps = 1.0 / (total_infer_time / total_samples)

            print("\n🏆 Official MK-UNet (BUSI dataset) 最终测试集成绩 🏆")
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