import os
import random
import time
import math
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

from main import HybridLoss
from main import calculate_metrics

# ==========================================
# 导入 LV-UNet
# ==========================================
from LV_UNet import LV_UNet


# ==========================================
# 6. UDIAT 数据集加载 (原图/GT结构)
# ==========================================
class UDIATDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths, self.mask_paths, self.is_train = image_paths, mask_paths, is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        if self.is_train:
            # 1. 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 2. 随机旋转
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # 3. 增加一点尺度缩放和平移 (超声图像中病灶大小差异很大)
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                translate = [int(random.uniform(-10, 10)), int(random.uniform(-10, 10))]
                image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0)
                mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # 【关键修复】加上 ImageNet Normalization
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
                # 兼容大小写差异（如 .PNG 和 .png）
                alt_f = f.replace('.PNG', '.png') if '.PNG' in f else f.replace('.png', '.PNG')
                alt_mask_path = os.path.join(gt_dir, alt_f)
                if os.path.exists(alt_mask_path):
                    img_p.append(img_path)
                    mask_p.append(alt_mask_path)

    print(f"✅ 成功找到 {len(img_p)} 对 UDIAT 图像和掩码。")
    return img_p, mask_p


# ==========================================
# 7. 主控台
# ==========================================
def main():
    # 🌟🌟🌟 控制面板 🌟🌟🌟
    MODE = "train"  # "train" 训练(微调) | "test" 直接评估
    DEEP_TRAINING = True  # 开启 LV-UNet 的深度训练特性
    data_dir = r"D:\PycharmProjects\data\UDIAT_Dataset_B"  # 你的 UDIAT 数据集根目录

    # 🚀 修改 1：对应上一份代码，读取 LV-UNet 在 BUSI 上的预训练权重
    pretrained_busi_path = "best_busi_lvunet.pth"
    save_path = "best_UDIAT_finetuned_lvunet.pth"  # 微调后保存的新权重名

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | ⚙️ Mode: {MODE} | Deep Training: {DEEP_TRAINING}")

    all_imgs, all_masks = get_udiat_paths(data_dir)
    if not all_imgs: return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total = len(all_imgs)
    t_size, v_size = int(0.8 * total), int(0.1 * total)

    # 🚀 修改 2：初始化 LV_UNet
    model = LV_UNet().to(device)

    if MODE == "train":
        if os.path.exists(pretrained_busi_path):
            print(f"🔄 正在加载 BUSI 预训练权重: {pretrained_busi_path}")
            model.load_state_dict(torch.load(pretrained_busi_path, map_location=device))
            print("✅ 预训练权重加载成功！开始基于 BUSI 先验知识进行微调 (Fine-tuning)。")
        else:
            print(f"⚠️ 警告: 未找到预训练权重 '{pretrained_busi_path}'，模型将从头开始训练！")

        train_loader = DataLoader(UDIATDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=8, shuffle=True)
        val_loader = DataLoader(
            UDIATDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False), batch_size=8,
            shuffle=False)

        criterion = HybridLoss()

        fine_tune_lr = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr, weight_decay=1e-4)

        num_epochs = 50
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0
        history_loss, history_val_dice = [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0

            # LV-UNet 特性: 动态深度训练 (基于 Cosine 函数调节激活参数)
            if DEEP_TRAINING:
                act_learn = 1 - math.cos(math.pi / 2 * epoch / num_epochs)
                try:
                    model.change_act(act_learn)
                except AttributeError:
                    pass

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                # 🚀 修改 3：LV_UNet 输出的是 logits，经过 sigmoid 转换为概率
                outputs = torch.sigmoid(model(images))
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # 验证评估
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    out = torch.sigmoid(model(images.to(device)))
                    val_dices.extend(calculate_metrics(out, masks.to(device))["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(
                f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                # 🚀 修改 4：保存微调后的 LV_UNet 最佳模型
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Saved] Val Dice: {best_val_dice:.4f} -> Saved to {save_path}")

        # 训练结束后静态绘制曲线图
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1);
        plt.plot(history_loss, marker='o');
        plt.title("Train Loss");
        plt.grid()
        plt.subplot(1, 2, 2);
        plt.plot(history_val_dice, marker='s', color='orange');
        plt.title("Val Dice");
        plt.grid()
        plt.tight_layout()
        plt.savefig('training_curve_LVUNet_UDIAT_Finetuned.png', dpi=150)
        print("✅ 训练完成，训练曲线已保存为 training_curve_LVUNet_UDIAT_Finetuned.png")

    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的测试集上评估...")
    test_loader = DataLoader(UDIATDataset(all_imgs[t_size + v_size:], all_masks[t_size + v_size:], False),
                             batch_size=1, shuffle=False)

    # 测试阶段：加载微调后保存的最新权重
    if not os.path.exists(save_path):
        print(f"❌ 找不到权重文件 {save_path}！请先运行 train 模式。")
        return

    model.load_state_dict(torch.load(save_path, map_location=device))

    # 🚀 修改 5：开启 LV-UNet 的重参数化推理部署模式 (提升推理速度)
    if DEEP_TRAINING:
        model.switch_to_deploy()
        print("⚡ 已开启 LV-UNet 重参数化部署模式 (switch_to_deploy)")

    model.eval()

    print("🔥 GPU 预热 (Warm-up)...")
    with torch.no_grad():
        for _ in range(5): model(torch.randn(1, 3, 256, 256).to(device))
    if torch.cuda.is_available(): torch.cuda.synchronize()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
    total_time, total_samples = 0.0, 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()

            # LV_UNet 推理
            outputs = torch.sigmoid(model(images))

            if torch.cuda.is_available(): torch.cuda.synchronize()

            total_time += (time.time() - t0)
            total_samples += images.size(0)

            m = calculate_metrics(outputs, masks)
            for k in test_res: test_res[k].extend(m[k])

    print("\n🏆 LV-UNet (UDIAT Finetuned) 测试集最终成绩 🏆")
    print(f"🔹 Dice : {np.mean(test_res['dice']):.4f} | IoU: {np.mean(test_res['iou']):.4f}")
    print(
        f"🔹 ACC  : {np.mean(test_res['acc']):.4f}  | Pre: {np.mean(test_res['pre']):.4f} | Rec: {np.mean(test_res['rec']):.4f}")
    print("-" * 50)
    print(f"⚡ 推理速度评估 (Device: {device}) ⚡")
    print(
        f"⏱️ 平均耗时: {(total_time / total_samples) * 1000:.2f} ms/img | FPS: {1.0 / (total_time / total_samples):.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()