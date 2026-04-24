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

# ==========================================
# 1. 从 Github 源码导入 MISSFormer
# ==========================================
# 请确保当前脚本与 MISSFormer 仓库中的 networks 文件夹在同一层级
from networks.MISSFormer import MISSFormer


class MISSFormerWrapper(nn.Module):
    """
    包装器：用于将 MISSFormer 的输出通过 Sigmoid 激活，
    以无缝适配下方针对二分类设计的 BCELoss (HybridLoss)。
    """

    def __init__(self, n_classes=1):
        super(MISSFormerWrapper, self).__init__()
        # 实例化 MISSFormer
        self.model = MISSFormer(num_classes=n_classes)

    def forward(self, x):
        logits = self.model(x)
        # 使用 sigmoid 激活以适配二分类概率输出
        return torch.sigmoid(logits)


# ==========================================
# 2. 原作者风格的损失函数与指标计算
# ==========================================
class AuthorStyleLoss(nn.Module):
    """
    完全遵循原作者 trainer.py 中的 Poly Loss 权重：0.4 * CE + 0.6 * Dice
    由于本任务是二分类，此处使用 BCE 替代 CE 进行等价转换。
    """

    def __init__(self):
        super(AuthorStyleLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # 1. 0.4 * BCE Loss (等价于二分类的 CrossEntropy)
        loss_bce = self.bce(pred, target)

        # 2. 0.6 * Dice Loss (采用原作者 utils.py 中的 Dice 计算逻辑: smooth=1e-5)
        smooth = 1e-5
        intersect = torch.sum(pred * target)
        y_sum = torch.sum(target * target)
        # 将原作者的 z_sum = torch.sum(pred * pred) 替换为标准的：
        dice = (2 * intersect + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)

        loss_dice = 1 - dice

        # 原作者组合比例
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
# 3. 数据集与增强加载 (尺寸修改为原作者的 224)
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

        # MISSFormer 特征图尺度与 224x224 强绑定
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224))

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # MISSFormer 原作使用 Normalize([0.5], [0.5])
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return image, TF.to_tensor(mask)


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
    # ==========================================
    # 1. 运行模式设置
    # ==========================================
    MODE = "train"  # 可选: "train" 或 "test"
    save_path = "best_busi_missformer.pth"

    data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 🔄 Current Mode: {MODE.upper()}")

    # ==========================================
    # 2. 数据集加载与划分
    # ==========================================
    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ 未找到数据！请检查路径。")
        exit()

    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

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
    # 3. 初始化 MISSFormer
    # ==========================================
    # ==========================================
    # 3. 初始化 MISSFormer
    # ==========================================
    model = MISSFormerWrapper(n_classes=1).to(device)

    # ==========================================
    # [新增] 3.5 加载 MiT-B1 ImageNet 预训练权重
    # ==========================================
    pretrained_path = r"./mit_b1.pth"  # ⚠️ 改成你实际下载的路径

    if os.path.exists(pretrained_path):
        print(f"🚀 正在加载 MiT Backbone 预训练权重: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')

        # 处理官方权重字典的键名兼容性
        # 官方权重的 key 可能带有 "backbone." 或 "default." 前缀，这里做简单清洗
        model_dict = model.model.backbone.state_dict()
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            # 去除可能存在的前缀干扰，匹配当前 backbone 的 key
            new_k = k.replace('backbone.', '').replace('head.', '')
            if new_k in model_dict and v.size() == model_dict[new_k].size():
                new_state_dict[new_k] = v

        # 将清洗后的权重加载到 MISSFormer 的 backbone 中 (strict=False 防止非关键层报错)
        model.model.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 成功加载了 {len(new_state_dict)} 个预训练张量！")
    else:
        print("⚠️ 警告：未找到预训练权重！模型将从头开始训练 (Train from scratch)，性能可能会非常差！")

    criterion = AuthorStyleLoss()

    # ==========================================
    # 4. 训练模式 (Train)
    # ==========================================
    # ==========================================
    # 4. 训练模式 (Train) - 针对 35 Epoch 的极速收敛策略
    # ==========================================
    if MODE == "train":
        num_epochs = 35
        best_val_dice = 0.0

        # ---------------------------------------------------------
        # 修改 1 & 2: 分离参数并使用 AdamW + 差分学习率
        # ---------------------------------------------------------
        backbone_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        # Backbone 采用较小学习率(1e-4)微调，Decoder 采用较大学习率(1e-3)快速学习
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4},
            {'params': decoder_params, 'lr': 1e-3}
        ], weight_decay=1e-4)

        # ---------------------------------------------------------
        # 修改 3: 使用余弦退火学习率策略 (Cosine Annealing)
        # ---------------------------------------------------------
        # T_max 设为总 iterations 数，让学习率在 35 epoch 内呈余弦曲线平滑下降至 eta_min
        max_iterations = num_epochs * len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-6)

        # 初始化绘图
        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax2.set_ylabel('Validation Dice', color='tab:blue')
        plt.title('Training Progress (AdamW + Cosine LR for 35 Epochs)')
        line_loss, = ax1.plot([], [], color='tab:red', marker='o', label='Train Loss')
        line_dice, = ax2.plot([], [], color='tab:blue', marker='s', label='Val Dice')
        ax1.legend([line_loss, line_dice], ['Train Loss', 'Val Dice'], loc='center right')
        history_loss, history_val_dice, epochs_list = [], [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0

            pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Train", leave=False)
            for i_batch, (images, masks) in enumerate(pbar_train):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()

                # 可选：加入梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # 每个 iteration 更新一次学习率
                scheduler.step()

                # 获取当前 Decoder 的学习率用于打印展示
                current_lr = optimizer.param_groups[1]['lr']

                train_loss_epoch += loss.item()
                pbar_train.set_postfix({'Loss': f"{loss.item():.4f}", 'LR(Dec)': f"{current_lr:.6f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # ... [之后的评估验证代码保持不变] ...

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_dices.extend(calculate_metrics(outputs, masks)["dice"])

            avg_val_dice = np.mean(val_dices)

            print(
                f"📉 Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Current LR: {current_lr:.6f}")

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

            # 保存最佳模型
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Model Saved] Val Dice: {best_val_dice:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        # 结束绘图
        plt.ioff()
        plt.savefig('missformer_author_settings_curve.png', dpi=150)
        print("📈 训练曲线已保存为 missformer_author_settings_curve.png")

    # ==========================================
    # 5. 测试模式 (Test) / 全指标评估
    # ==========================================
    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print("🚀 开始在完全未见的【测试集 (Test Set)】上评估 MISSFormer 最终性能...")

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

            print("\n🏆  MISSFormer (BUSI dataset) 最终测试集成绩 🏆")
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