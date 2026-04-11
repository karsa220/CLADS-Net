import os
import random
import time
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
import pywt


# ==========================================
# 1. 小波变换基础算子 (优化版：引入反射填充避免边缘伪影)
# ==========================================
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad_h = filters.shape[2] // 2 - 1
    pad_w = filters.shape[3] // 2 - 1
    # 反射填充 (Reflection Padding)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    x = F.conv2d(x, filters, stride=2, groups=c)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad_h = filters.shape[2] // 2 - 1
    pad_w = filters.shape[3] // 2 - 1
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=(pad_h, pad_w))
    return x


# ==========================================
# 2. 核心模块：带高频门控与LL大核增强的 SFEB
# ==========================================
class SFEB_Advanced(nn.Module):
    def __init__(self, in_channels, wt_type='db2'):
        super(SFEB_Advanced, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=True)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=True)

        self.ll_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.hf_dir_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]).view(1, 3, 1, 1))
        self.hf_mix = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 3),
            nn.ReLU(inplace=True)
        )
        self.hf_spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(in_channels // 4, 1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        local_feat = self.local_conv(x)
        curr_shape = local_feat.shape
        pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
        local_feat_padded = F.pad(local_feat, pads) if sum(pads) > 0 else local_feat

        wt_out = wavelet_transform(local_feat_padded, self.wt_filter)
        ll, hl, lh, hh = wt_out[:, :, 0], wt_out[:, :, 1], wt_out[:, :, 2], wt_out[:, :, 3]

        ll_enhanced = self.ll_enhance(ll)

        hf_stack = torch.stack([hl, lh, hh], dim=1) * self.hf_dir_weights.unsqueeze(-1)
        b, _, c, h_f, w_f = hf_stack.shape
        hf_mixed = self.hf_mix(hf_stack.view(b, 3 * c, h_f, w_f))

        hf_mask = self.hf_spatial_attention(torch.cat([torch.mean(hf_mixed, dim=1, keepdim=True),
                                                       torch.max(hf_mixed, dim=1, keepdim=True)[0]], dim=1))
        hl_g, lh_g, hh_g = torch.chunk(hf_mixed * hf_mask, 3, dim=1)

        freq_feat = inverse_wavelet_transform(torch.stack([ll_enhanced, hl_g, lh_g, hh_g], dim=2), self.iwt_filter)
        freq_feat = freq_feat[:, :, :curr_shape[2], :curr_shape[3]] + local_feat

        fused_feat = self.fusion_conv(freq_feat)
        return fused_feat * self.channel_att(self.gap(fused_feat)) + identity


# ==========================================
# 3. 注意力卷积块
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(inplace=True),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.conv1(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))


class AttentionConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))
        self.ca, self.sa = ChannelAttention(out_ch), SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        return x * self.ca(x) * self.sa(x)


class BottleneckSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, h * w)
        attention = F.softmax(torch.bmm(proj_query, proj_key) / ((c // 8) ** 0.5), dim=-1)
        out = torch.bmm(self.value_conv(x).view(b, -1, h * w), attention.permute(0, 2, 1))
        return self.gamma * out.view(b, c, h, w) + x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, 1, 0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x), inplace=True))


# ==========================================
# 4. 网络主体
# ==========================================
class HANet_Final(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 256, 512, 1024

        self.sfeb2 = SFEB_Advanced(self.ch2, wt_type='db2')
        self.sfeb3 = SFEB_Advanced(self.ch3, wt_type='db2')
        self.bottleneck_attn = BottleneckSelfAttention(in_channels=1024)
        self.ag1 = AttentionGate(F_g=128, F_l=self.ch1, F_int=64)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = AttentionConvBlock(1024 + self.ch4, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = AttentionConvBlock(512 + self.ch3, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = AttentionConvBlock(256 + self.ch2, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = AttentionConvBlock(128 + self.ch1, 64)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec0 = AttentionConvBlock(64, 32)

        self.out3 = nn.Conv2d(256, 1, kernel_size=1)
        self.out2 = nn.Conv2d(128, 1, kernel_size=1)
        self.out1 = nn.Conv2d(64, 1, kernel_size=1)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x)))
        e2 = self.encoder.denseblock1(self.encoder.pool0(e0))
        e3 = self.encoder.denseblock2(self.encoder.transition1(e2))
        e4 = self.encoder.denseblock3(self.encoder.transition2(e3))
        b = self.bottleneck_attn(self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e4))))

        skip2, skip3 = self.sfeb2(e2), self.sfeb3(e3)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))

        up_d2 = self.up1(d2)
        d1 = self.dec1(torch.cat([up_d2, self.ag1(g=up_d2, x=e0)], dim=1))

        out = torch.sigmoid(self.final_conv(self.dec0(self.up0(d1))))

        if self.training:
            return out, torch.sigmoid(self.out1(d1)), torch.sigmoid(self.out2(d2)), torch.sigmoid(self.out3(d3))
        return out


# ==========================================
# 5. 混合损失函数与评估指标
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.weight_bce, self.weight_dice = weight_bce, weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        pred_flat, target_flat = pred.view(pred.size(0), -1), target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_score = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        return (self.weight_bce * bce_loss) + (self.weight_dice * (1.0 - dice_score).mean())


def calculate_metrics(pred, target, smooth=1e-5):
    pred_bin, target_bin = (pred > 0.5).float().view(pred.size(0), -1), target.float().view(target.size(0), -1)
    tp = (pred_bin * target_bin).sum(dim=1)
    fp = (pred_bin * (1 - target_bin)).sum(dim=1)
    fn = ((1 - pred_bin) * target_bin).sum(dim=1)
    tn = ((1 - pred_bin) * (1 - target_bin)).sum(dim=1)

    return {
        "dice": ((2. * tp + smooth) / (2. * tp + fp + fn + smooth)).tolist(),
        "iou": ((tp + smooth) / (tp + fp + fn + smooth)).tolist(),
        "acc": ((tp + tn + smooth) / (tp + fp + fn + tn + smooth)).tolist(),
        "pre": ((tp + smooth) / (tp + fp + smooth)).tolist(),
        "rec": ((tp + smooth) / (tp + fn + smooth)).tolist()
    }


# ==========================================
# 6. UDIAT 数据集加载 (原图/GT结构)
# ==========================================
class UDIATDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths, self.mask_paths, self.is_train = image_paths, mask_paths, is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 统一转RGB，避免灰度图通道数不匹配Densnet要求
        image = TF.resize(Image.open(self.image_paths[idx]).convert('RGB'), (256, 256))
        mask = TF.resize(Image.open(self.mask_paths[idx]).convert('L'), (256, 256))

        if self.is_train:
            if random.random() > 0.5: image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() > 0.5: image, mask = TF.vflip(image), TF.vflip(mask)
            angle = random.uniform(-15, 15)
            image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)

        return TF.to_tensor(image), TF.to_tensor(mask)


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
    MODE = "test"  # "train" 训练 | "test" 直接评估
    data_dir = r"D:\PycharmProjects\data\UDIAT_Dataset_B"  # 你的 UDIAT 数据集根目录
    save_path = "best_HANet_UDIAT.pth"

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

    model = HANet_Final().to(device)

    if MODE == "train":
        train_loader = DataLoader(UDIATDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=8, shuffle=True)
        val_loader = DataLoader(
            UDIATDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False), batch_size=8,
            shuffle=False)

        criterion = HybridLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        num_epochs = 40
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

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

                # 深度监督 Loss 计算
                if isinstance(outputs, tuple):
                    out, out1, out2, out3 = outputs
                    loss_main = criterion(out, masks)
                    ts = masks.shape[2:]
                    loss_aux = sum(
                        criterion(F.interpolate(o, size=ts, mode='bilinear', align_corners=True), masks) for o in
                        [out1, out2, out3])
                    loss = loss_main + 0.4 * loss_aux
                else:
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
                    out = model(images.to(device))
                    # 防御性编程：万一 eval 模式下返回了 tuple，取主输出
                    if isinstance(out, tuple): out = out[0]
                    val_dices.extend(calculate_metrics(out, masks.to(device))["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(
                f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Saved] Val Dice: {best_val_dice:.4f}")

        # 训练结束后静态绘制曲线图 (避免环境 GUI 兼容性报错)
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
        plt.savefig('training_curve_HANet_UDIAT.png', dpi=150)
        print("✅ 训练完成，训练曲线已保存为 training_curve_HANet_UDIAT.png")


    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的测试集上评估...")
    test_loader = DataLoader(UDIATDataset(all_imgs[t_size + v_size:], all_masks[t_size + v_size:], False),
                             batch_size=1, shuffle=False)

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
            if torch.cuda.is_available(): torch.cuda.synchronize()

            total_time += (time.time() - t0)
            total_samples += images.size(0)

            if isinstance(outputs, tuple): outputs = outputs[0]
            m = calculate_metrics(outputs, masks)
            for k in test_res: test_res[k].extend(m[k])

    print("\n🏆 HANet_Final (UDIAT) 测试集最终成绩 🏆")
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