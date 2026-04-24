import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

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
# 2. UDIAT 数据集加载 (原图/GT结构)
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

        # 【关键修复1】加上 ImageNet Normalization
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 【关键修复2】适配官方 CrossEntropyLoss 的类别标签格式 (H, W) -> LongTensor
        mask = torch.where(mask > 0.5, 1, 0).squeeze(0).long()

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
# 3. 主控台
# ==========================================
def main():
    # 🌟🌟🌟 控制面板 🌟🌟🌟
    MODE = "train"  # "train" 训练(微调) | "test" 直接评估
    data_dir = r"D:\PycharmProjects\data\UDIAT_Dataset_B"  # 你的 UDIAT 数据集根目录

    # 🚀 定义预训练权重和新权重的路径
    pretrained_busi_path = "best_busi_official.pth"  # 确保这里加载的是用官方代码训练出来的 BUSI 权重
    save_path = "best_UDIAT_finetuned.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | ⚙️ Mode: {MODE}")

    all_imgs, all_masks = get_udiat_paths(data_dir)
    if not all_imgs: return

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total = len(all_imgs)
    t_size, v_size = int(0.8 * total), int(0.1 * total)

    # ==========================================
    # 官方模型初始化
    # ==========================================
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 2  # 类别数：背景 + 肿块
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))  # 输入分辨率为 256
    model = ViT_seg(config_vit, img_size=256, num_classes=2).to(device)

    # 官方损失函数
    ce_loss = nn.CrossEntropyLoss()
    dice_loss_func = DiceLoss(2)

    if MODE == "train":
        # 🚀 在开启训练前，加载之前使用官方代码训练出的 BUSI 预训练权重
        if os.path.exists(pretrained_busi_path):
            print(f"🔄 正在加载 BUSI 预训练权重: {pretrained_busi_path}")
            model.load_state_dict(torch.load(pretrained_busi_path, map_location=device))
            print("✅ 预训练权重加载成功！开始基于 BUSI 先验知识进行微调 (Fine-tuning)。")
        else:
            print(
                f"⚠️ 警告: 未找到预训练权重 '{pretrained_busi_path}'，模型将从头开始训练！(如已有预训练文件请确认文件名)")

        train_loader = DataLoader(UDIATDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=8, shuffle=True)
        val_loader = DataLoader(
            UDIATDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False), batch_size=8,
            shuffle=False)

        # 🚀 官方优化器设置：SGD, 带有动量和衰减
        # 因为是微调，我们可以使用比 0.01 略低的基础学习率 (比如 0.001)
        base_lr = 0.001
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

        num_epochs = 50
        max_iterations = num_epochs * len(train_loader)
        iter_num = 0

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
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{lr_:.6f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # 验证评估
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    out = model(images.to(device))
                    # 转换模型输出 logits 为具体的类别掩码 (B, H, W)
                    preds = torch.argmax(torch.softmax(out, dim=1), dim=1)
                    val_dices.extend(calculate_metrics(preds, masks.to(device))["dice"])

            avg_val_dice = np.mean(val_dices)
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {lr_:.6f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
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
        plt.savefig('training_curve_Official_UDIAT_Finetuned.png', dpi=150)
        print("✅ 训练完成，训练曲线已保存为 training_curve_Official_UDIAT_Finetuned.png")

    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的测试集上评估...")
    test_loader = DataLoader(UDIATDataset(all_imgs[t_size + v_size:], all_masks[t_size + v_size:], False),
                             batch_size=1, shuffle=False)

    # 测试阶段：加载微调后保存的最新权重
    if not os.path.exists(save_path):
        print(f"❌ 找不到权重文件 {save_path}！请先运行 train 模式。")
        return

    model.load_state_dict(torch.load(save_path, map_location=device))
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

            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # 提取推理结果

            if torch.cuda.is_available(): torch.cuda.synchronize()

            total_time += (time.time() - t0)
            total_samples += images.size(0)

            m = calculate_metrics(preds, masks)
            for k in test_res: test_res[k].extend(m[k])

    print("\n🏆 Official TransUNet (UDIAT Finetuned) 测试集最终成绩 🏆")
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