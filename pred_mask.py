import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from CLADS_Net_busi import HANet_Final


# 如果你的 HANet_Final 模型代码保存在其他文件 (比如 model.py)，请在这里导入
# from model import HANet_Final
# 为了脚本能直接运行，请确保 HANet_Final 的类定义在这个脚本中或者被正确导入

def calculate_single_dice(pred_mask, true_mask, smooth=1e-5):
    """计算单张图像的 Dice 分数"""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    intersection = np.sum(pred_flat * true_flat)
    return (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(true_flat) + smooth)


def visualize_prediction(model_path, image_path, mask_path):
    # 1. 基础设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 2. 初始化模型并加载权重
    # 注意：这里需要确保 HANet_Final 类在作用域内
    model = HANet_Final().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到权重文件: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 必须设置为评估模式

    # 3. 加载与预处理图像
    print(f"📂 加载图像: {os.path.basename(image_path)}")
    img_pil = Image.open(image_path).convert('RGB')
    mask_pil = Image.open(mask_path).convert('L')

    # 保持与训练时一致的 256x256 分辨率
    img_resized = TF.resize(img_pil, (256, 256))
    mask_resized = TF.resize(mask_pil, (256, 256))

    # 转换为 Tensor 并且增加 Batch 维度 (1, C, H, W)
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
    mask_tensor = TF.to_tensor(mask_resized).squeeze().numpy()  # [256, 256] 真实的 numpy 数组，用于计算
    mask_binary = (mask_tensor > 0.5).astype(np.float32)

    # 4. 模型推理
    with torch.no_grad():
        # 在 eval 模式下，你的代码只返回 out
        pred = model(img_tensor)
        # 将 [1, 1, 256, 256] 的概率图转为 [256, 256] 的二值化 numpy 数组
        pred_prob = pred.squeeze().cpu().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.float32)

    # 5. 计算当前图像的 Dice 分数
    dice_score = calculate_single_dice(pred_binary, mask_binary)
    print(f"🎯 该图像的预测 Dice 分数: {dice_score:.4f}")

    # 6. 可视化绘图
    img_display = np.array(img_resized)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 图 1：原图
    axes[0].imshow(img_display)
    axes[0].set_title('Original Ultrasound', fontsize=14)
    axes[0].axis('off')

    # 图 2：真实掩膜 (Ground Truth)
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=14)
    axes[1].axis('off')

    # 图 3：预测掩膜 (Prediction)
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title(f'Prediction (Dice: {dice_score:.4f})', fontsize=14)
    axes[2].axis('off')

    # 图 4：原图与预测结果的叠加 (Overlay)
    # 用红色高亮预测区域，绿色轮廓高亮真实区域 (或者直接用半透明掩膜覆盖)
    axes[3].imshow(img_display)
    # 创建一个红色的透明层用于显示预测结果
    red_mask = np.zeros_like(img_display)
    red_mask[:, :, 0] = 255  # 红色通道拉满
    # 当 pred_binary 为 1 时，显示 0.4 透明度的红色
    alpha_layer = pred_binary * 0.4
    axes[3].imshow(red_mask, alpha=alpha_layer)
    # 用等高线画出 Ground Truth 的边缘（绿色）
    axes[3].contour(mask_binary, levels=[0.5], colors='lime', linewidths=2)

    axes[3].set_title('Overlay (Red: Pred, Green: True Edge)', fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ================= 参数配置区 =================
    # 1. 填入你训练好的权重路径
    WEIGHT_PATH = "best_model_base.pth"

    # 2. 填入测试集中某一张图片的路径和对应的掩膜路径
    # 请确保这张图片模型在训练时绝对没有见过
    TEST_IMAGE_PATH = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT\malignant\malignant (5).png"
    TEST_MASK_PATH = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT\malignant\malignant (5)_mask.png"
    # ==============================================

    visualize_prediction(WEIGHT_PATH, TEST_IMAGE_PATH, TEST_MASK_PATH)