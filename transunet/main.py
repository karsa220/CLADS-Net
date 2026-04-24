import os
import random
import time
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

# ==========================================
# 导入官方 TransUNet 依赖
# ==========================================
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import DiceLoss


# ==========================================
# 1. 评估指标计算 (适配官方多通道输出)
# ==========================================
def calculate_metrics(pred_bin, target, smooth=1e-5):
    # pred_bin 和 target 的 shape 均为 (B, H, W)
    pred_flat = pred_bin.view(pred_bin.size(0), -1).float()
    target_flat = target.view(target.size(0), -1).float()

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
# 2. 数据集加载 (调整 Mask 输出格式以适配官方 Loss)
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

        # Mask 转换为 (256, 256) 的 LongTensor 类别索引 (0 为背景，1 为目标)
        mask_tensor = TF.to_tensor(mask)
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).squeeze(0).long()

        return TF.to_tensor(image), mask_tensor


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
# 3. 主控台
# ==========================================
def main():
    MODE = "train"
    save_path = "best_busi_official.pth"
    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 🔄 Current Mode: {MODE.upper()}")

    # --- 数据集加载与划分 ---
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

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ==========================================
    # 官方模型初始化
    # ==========================================
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']  # 使用官方 R50+ViT-B_16 配置
    config_vit.n_classes = 2  # 背景(0) + 肿块(1)
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))  # 适配 256x256 输入

    model = ViT_seg(config_vit, img_size=256, num_classes=2).to(device)

    # 【可选】加载官方 ImageNet21k 预训练权重
    # 如果你下载了预训练权重，请取消注释下面两行并修改路径：
    # pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    # model.load_from(weights=np.load(pretrained_path))

    # ==========================================
    # 官方损失函数
    # ==========================================
    ce_loss = nn.CrossEntropyLoss()
    dice_loss_func = DiceLoss(2)

    # ==========================================
    # 训练模式
    # ==========================================
    if MODE == "train":
        # 官方优化器设置：SGD, lr=0.01, momentum=0.9, weight_decay=1e-4
        base_lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

        num_epochs = 35
        max_iterations = num_epochs * len(train_loader)
        iter_num = 0
        best_val_dice = 0.0

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Official TransUNet Training Progress')
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

                outputs = model(images)  # 官方模型输出形状: (B, 2, H, W)

                # 官方 Loss 组合：0.5 CE + 0.5 Dice
                loss_ce = ce_loss(outputs, masks)
                loss_dice = dice_loss_func(outputs, masks, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

                loss.backward()
                optimizer.step()

                # 官方 Poly 学习率衰减策略
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num += 1

                train_loss_epoch += loss.item()
                pbar_train.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{lr_:.6f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # --- 验证阶段 ---
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    # 将模型输出的 logits 转为预测类别
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                    val_dices.extend(calculate_metrics(preds, masks)["dice"])

            avg_val_dice = np.mean(val_dices)

            print(
                f"📉 Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {lr_:.6f}")

            # 绘图更新
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
                print(f"🌟 [New Best Model Saved] Val Dice: {best_val_dice:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        plt.ioff()
        plt.savefig('official_transunet_curve.png', dpi=150)
        print("📈 官方配置训练曲线已保存为 official_transunet_curve.png")

    # ==========================================
    # 测试模式
    # ==========================================
    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print("🚀 开始在测试集评估 Official TransUNet ...")

        if not os.path.exists(save_path):
            print(f"❌ 找不到权重文件: {save_path}")
        else:
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            with torch.no_grad():
                for _ in range(10): _ = model(dummy_input)

            total_infer_time = 0.0
            total_samples = 0

            with torch.no_grad():
                for images, masks in tqdm(test_loader, desc="Testing"):
                    images, masks = images.to(device), masks.to(device)

                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    start_time = time.time()

                    outputs = model(images)
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # 提取预测类别

                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    end_time = time.time()

                    total_infer_time += (end_time - start_time)
                    total_samples += images.size(0)

                    metrics = calculate_metrics(preds, masks)
                    for k in test_res.keys():
                        test_res[k].extend(metrics[k])

            avg_time_per_image = (total_infer_time / total_samples) * 1000
            fps = 1.0 / (total_infer_time / total_samples)
            print("\n🏆 Official TransUNet (BUSI dataset) 最终成绩 🏆")
            print(f"🔹 Dice     : {np.mean(test_res['dice']):.4f}")
            print(f"🔹 IoU      : {np.mean(test_res['iou']):.4f}")
            print(f"🔹 ACC      : {np.mean(test_res['acc']):.4f}")
            print(f"🔹 Precision: {np.mean(test_res['pre']):.4f}")
            print(f"🔹 Recall   : {np.mean(test_res['rec']):.4f}")
            print("-" * 50)
            print(f"⏱️ 平均耗时 : {avg_time_per_image:.2f} ms / image")
            print(f"🚀 F P S    : {fps:.2f} frames / second")


if __name__ == "__main__":
    main()