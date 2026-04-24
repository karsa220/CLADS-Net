import os
import random
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

# ==========================================
# 1. 模型导入 (请根据实际目录结构调整)
# ==========================================
from CTNet.main import CTNet
from CMUNet.main import CMUNet
from hiformer.main import HiFormer
from PVT_EMCAD_B2.main import PVT_EMCAD_B2
from CLADSNet_busi import CLADS_Net


# ==========================================
# 2. 辅助函数：获取数据集路径 & 获取掩膜路径
# ==========================================
def get_dataset_paths(data_dir):
    """用于获取 BUSI 的全部图像和掩膜路径"""
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


def get_mask_path(img_path):
    """用于为固定的 UDIAT 原图精准匹配对应的 GT 掩膜"""
    dir_name = os.path.dirname(img_path)
    base_name = os.path.basename(img_path)

    if 'original' in dir_name:
        name, ext = os.path.splitext(base_name)
        udiat_mask = os.path.join(dir_name.replace('original', 'GT'), base_name)
        if os.path.exists(udiat_mask):
            return udiat_mask

        # 兼容后缀大小写差异 (.png / .PNG)
        alt_ext = '.PNG' if ext == '.png' else '.png'
        alt_mask = os.path.join(dir_name.replace('original', 'GT'), name + alt_ext)
        if os.path.exists(alt_mask):
            return alt_mask

    raise FileNotFoundError(f"❌ 找不到对应的 UDIAT 掩膜文件: {img_path}")


# ==========================================
# 3. 可视化主流程
# ==========================================
def visualize_hybrid_selection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- 目录配置 ---
    busi_data_dir = r"D:\PycharmProjects\data\Dataset_BUSI_with_GT"
    weights_dir = r"D:\PycharmProjects\CLADS-Net"

    # 🚨 UDIAT 固定的两张图片路径
    FIXED_UDIAT_PATHS = [
        r"D:\PycharmProjects\data\UDIAT_Dataset_B\original\000123.png",
        r"D:\PycharmProjects\data\UDIAT_Dataset_B\original\000089.png"
    ]

    # --- 模型排布顺序 (决定了列的顺序，Ours 放最前面) ---
    model_configs = {
        "CLADSNet (Ours)": CLADS_Net(embed_dim=256),
        "CTNet": CTNet(),
        "CMUNet": CMUNet(),
        "HiFormer": HiFormer(),
        "PVT_EMCAD_B2": PVT_EMCAD_B2(),
    }

    # ---------------------------------------------------------
    # 批量加载模型权重到内存
    # ---------------------------------------------------------
    print("⏳ 正在加载模型权重...")
    loaded_models_busi = {}
    loaded_models_udiat = {}

    for model_name, model_instance in model_configs.items():
        folder_name = model_name if os.path.exists(
            os.path.join(weights_dir, model_name)) else model_name.lower().replace(" ", "_").replace("(", "").replace(
            ")", "")

        # 1. 加载 BUSI 权重
        busi_weight = os.path.join(weights_dir, folder_name, "best_busi.pth")
        if os.path.exists(busi_weight):
            m_busi = copy.deepcopy(model_instance)
            m_busi.load_state_dict(torch.load(busi_weight, map_location=device))
            m_busi = m_busi.to(device).eval()
            loaded_models_busi[model_name] = m_busi

        # 2. 加载 UDIAT 权重
        udiat_weight = os.path.join(weights_dir, folder_name, "best_UDIAT_finetuned.pth")
        if os.path.exists(udiat_weight):
            m_udiat = copy.deepcopy(model_instance)
            m_udiat.load_state_dict(torch.load(udiat_weight, map_location=device))
            m_udiat = m_udiat.to(device).eval()
            loaded_models_udiat[model_name] = m_udiat

    # ---------------------------------------------------------
    # 数据提取策略 (BUSI 测试集 + UDIAT 固定值)
    # ---------------------------------------------------------
    print("🎯 正在提取评估图片 (BUSI: 测试集随机 | UDIAT: 绝对指定)...")

    # 1. BUSI 提取 (严格从最后的 10% 测试集抽取)
    all_imgs_busi, all_masks_busi = get_dataset_paths(busi_data_dir)
    combined_busi = list(zip(all_imgs_busi, all_masks_busi))
    random.seed(42)  # 固定随机种子以对齐训练集切分
    random.shuffle(combined_busi)
    test_idx_busi = int(0.9 * len(combined_busi))  # 后 10% 为测试集
    test_busi = combined_busi[test_idx_busi:]

    # 从中随机挑选 2 张 (可通过修改这里的 seed 来换不同的测试集图片)
    random.seed(9)
    selected_busi_pairs = random.sample(test_busi, 2)

    # 2. UDIAT 提取 (固定路径)
    selected_udiat_pairs = [
        (FIXED_UDIAT_PATHS[0], get_mask_path(FIXED_UDIAT_PATHS[0])),
        (FIXED_UDIAT_PATHS[1], get_mask_path(FIXED_UDIAT_PATHS[1]))
    ]

    # 组装最终的 4 行数据
    selected_pairs = selected_busi_pairs + selected_udiat_pairs

    # --- 创建图表 (4行 x 7列) ---
    num_rows = len(selected_pairs)
    num_cols = 2 + len(model_configs)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    col_titles = ["Ultrasound Image", "Ground Truth"] + list(model_configs.keys())

    print("🎨 开始绘制对比图像...")
    for row_idx, (img_path, mask_path) in enumerate(selected_pairs):
        # 判断当前行使用的是哪套权重
        dataset_type = "busi" if row_idx < 2 else "udiat"
        current_models = loaded_models_busi if dataset_type == "busi" else loaded_models_udiat

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        gt_mask = Image.open(mask_path).convert('L')

        image = TF.resize(image, (256, 256))
        gt_mask = TF.resize(gt_mask, (256, 256))

        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
        gt_numpy = (np.array(gt_mask) > 127).astype(np.uint8)  # GT 二值化黑白

        # 绘制第 1 列: 原图
        ax_img = axes[row_idx, 0]
        ax_img.imshow(image)
        ax_img.axis('off')
        if row_idx == 0: ax_img.set_title(col_titles[0], fontsize=16, fontweight='bold', pad=10)

        # 绘制第 2 列: Ground Truth 掩膜
        ax_gt = axes[row_idx, 1]
        ax_gt.imshow(gt_numpy, cmap='gray')
        ax_gt.axis('off')
        if row_idx == 0: ax_gt.set_title(col_titles[1], fontsize=16, fontweight='bold', pad=10)

        # 绘制后续列: 各个模型的预测掩膜
        col_idx = 2
        for model_name in model_configs.keys():
            ax_model = axes[row_idx, col_idx]
            ax_model.axis('off')

            # 设置标题 (我们的模型标题加粗绿色)
            if row_idx == 0:
                color = 'green' if 'Ours' in model_name else 'black'
                ax_model.set_title(col_titles[col_idx], fontsize=16, fontweight='bold', color=color, pad=10)

            # 模型推理
            if model_name in current_models:
                model = current_models[model_name]
                with torch.no_grad():
                    output = model(img_tensor)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # 智能 Sigmoid 检测
                    if output.max() <= 1.0 and output.min() >= 0.0:
                        prob = output.squeeze().cpu().numpy()
                    else:
                        prob = torch.sigmoid(output).squeeze().cpu().numpy()

                    # 阈值化为纯黑白掩膜 (0: 黑色背景, 1: 白色肿瘤)
                    pred_mask = (prob > 0.5).astype(np.uint8)

                ax_model.imshow(pred_mask, cmap='gray')
            else:
                # 缺失权重占位
                ax_model.text(0.5, 0.5, 'Weights\nNot Found', ha='center', va='center', color='red', fontsize=12)

            col_idx += 1

    # 紧密排布
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # --- 保存和展示 ---
    save_fig_path = "segmentation_comparison_hybrid_paper.png"
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ 绘图完成，论文级排版图像已保存至 {save_fig_path}")
    plt.show()


if __name__ == "__main__":
    visualize_hybrid_selection()