import re

def sort_and_filter_models(input_text):
    # 1. 预处理：按 "# 数字" 进行切分
    chunks = re.split(r'\n(?=# \d+)', input_text.strip())

    model_data = []

    for chunk in chunks:
        # 提取模型名称
        name_match = re.search(r'# \d+ (.*)', chunk)
        if not name_match:
            continue
        model_name = name_match.group(1).split('\n')[0].strip()

        # 【核心修复部分】
        # 正则逻辑：匹配一整行奖杯标题，以及紧跟其后的所有 🔹 开头的指标行
        # 这样能自动滤除段落间夹杂的文字备注（例如"下面都是busi训练..."）
        sub_blocks = re.findall(r'(🏆[^\n]*🏆\n*(?:🔹[^\n]*\n*)+)', chunk)

        if len(sub_blocks) >= 2:
            # 提取 Dice 数值用于排序 (兼容 "Dice :" 和 "Dice (F1):")
            d1 = re.search(r'Dice.*?: (\d+\.\d+)', sub_blocks[0])
            d2 = re.search(r'Dice.*?: (\d+\.\d+)', sub_blocks[1])

            if d1 and d2:
                model_data.append({
                    'name': model_name,
                    'busi_content': sub_blocks[0].strip(),
                    'udiat_content': sub_blocks[1].strip(),
                    'dice1': float(d1.group(1)),
                    'dice2': float(d2.group(1))
                })

    # 2. 生成 BUSI 专用文档 (按 dice1 降序)
    sorted_busi = sorted(model_data, key=lambda x: x['dice1'], reverse=True)
    busi_output = "=== 按第一个 DICE (BUSI) 降序排序 - 仅保留 BUSI 成绩 ===\n\n"
    for i, m in enumerate(sorted_busi, 1):
        busi_output += f"# {i} {m['name']}\n\n{m['busi_content']}\n\n"
        busi_output += "-" * 50 + "\n\n"

    # 3. 生成 UDIAT 专用文档 (按 dice2 降序)
    sorted_udiat = sorted(model_data, key=lambda x: x['dice2'], reverse=True)
    udiat_output = "=== 按第二个 DICE (UDIAT) 降序排序 - 仅保留 UDIAT 成绩 ===\n\n"
    for i, m in enumerate(sorted_udiat, 1):
        udiat_output += f"# {i} {m['name']}\n\n{m['udiat_content']}\n\n"
        udiat_output += "-" * 50 + "\n\n"

    return busi_output, udiat_output


# 原始数据
raw_md = """
# 1 CLADS-Net (Ours) 


🏆 CLADS-Net（busi） (35EPOCH)最终测试集成绩 🏆
🔹 Dice     : 0.8205
🔹 IoU      : 0.7432
🔹 ACC      : 0.9719
🔹 Precision: 0.8235
🔹 Recall   : 0.8320

然后busi权重在udiat数据集微调50epoch


🏆 CLADSNet (UDIAT) 测试集最终成绩 🏆
🔹 Dice : 0.8845 | IoU: 0.8024
🔹 ACC  : 0.9856  | Pre: 0.8812 | Rec: 0.9127





下面都是 busi 训练35个epoch，然后busi权重在udiat微调50epoch
# 2 UNet

🏆  UNet (BUSI dataset) 最终测试集成绩 🏆
🔹 Dice     : 0.6780
🔹 IoU      : 0.5942
🔹 ACC      : 0.9515
🔹 Precision: 0.7699
🔹 Recall   : 0.7171

🏆 UNet (UDIAT dataset) 测试集最终成绩 🏆
🔹 Dice : 0.8209 | IoU: 0.7364
🔹 ACC  : 0.9876  | Pre: 0.8369 | Rec: 0.8837
# 3 TransUNet

🏆 Official TransUNet (BUSI dataset) 最终成绩 🏆
🔹 Dice     : 0.7294
🔹 IoU      : 0.6513
🔹 ACC      : 0.9616
🔹 Precision: 0.8176
🔹 Recall   : 0.7119
🏆 TransUNet (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.8187 | IoU: 0.7236
🔹 ACC  : 0.9874  | Pre: 0.8222 | Rec: 0.8820
--------------------------------------------------
# 4 SegFormer

🏆 SegFormer (BUSI dataset) 最终测试集成绩 🏆
🔹 Dice     : 0.7936
🔹 IoU      : 0.7077
🔹 ACC      : 0.9634
🔹 Precision: 0.8041
🔹 Recall   : 0.8076


🏆 SegFormer (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.8055 | IoU: 0.7152
🔹 ACC  : 0.9805  | Pre: 0.7896 | Rec: 0.8800


# 5 SwinUNet
🏆 SwinUNet (BUSI dataset)  最终测试集成绩 🏆
🔹 Dice     : 0.7363
🔹 IoU      : 0.6371
🔹 ACC      : 0.9624
🔹 Precision: 0.7938
🔹 Recall   : 0.7258

🏆 Official Swin-UNet (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.7852 | IoU: 0.6678
🔹 ACC  : 0.9831  | Pre: 0.7734 | Rec: 0.8684
--------------------------------------------------


# 6 UNet++
🏆  UNet++ (BUSI dataset)  最终测试集成绩 🏆
🔹 Dice     : 0.7807
🔹 IoU      : 0.6964
🔹 ACC      : 0.9653
🔹 Precision: 0.8009
🔹 Recall   : 0.7893

🏆 UNet++ (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.8285 | IoU: 0.7530
🔹 ACC  : 0.9888  | Pre: 0.8781 | Rec: 0.8620
--------------------------------------------------

# 7 MISSFormer

🏆  MISSFormer (BUSI dataset) 最终测试集成绩 🏆
🔹 Dice     : 0.7063
🔹 IoU      : 0.6102
🔹 ACC      : 0.9550
🔹 Precision: 0.7279
🔹 Recall   : 0.7457
🏆 MISSFormer (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice     : 0.8179
🔹 IoU      : 0.7232
🔹 ACC      : 0.9845
🔹 Precision: 0.8341
🔹 Recall   : 0.8771

# 8 LV-UNet
🏆 LV-UNet (BUSI dataset) 最终测试集成绩 🏆
🔹 Dice     : 0.7417
🔹 IoU      : 0.6331
🔹 ACC      : 0.9428
🔹 Precision: 0.7023
🔹 Recall   : 0.8509
🏆 LV-UNet (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.6072 | IoU: 0.4680
🔹 ACC  : 0.9445  | Pre: 0.5295 | Rec: 0.8552
--------------------------------------------------

# 9 CMU-Net
🏆  CMU-Net (BUSI dataset) 最终测试集成绩 🏆
🔹 Dice     : 0.7684
🔹 IoU      : 0.6807
🔹 ACC      : 0.9579
🔹 Precision: 0.7857
🔹 Recall   : 0.7910

🏆 CMU-Net (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.8216 | IoU: 0.7238
🔹 ACC  : 0.9865  | Pre: 0.8460 | Rec: 0.8595
--------------------------------------------------

# 10 MK-UNet

🏆 MK-UNet (BUSI dataset) 最终测试集成绩 🏆
🔹 Dice     : 0.7914
🔹 IoU      : 0.7157
🔹 ACC      : 0.9675
🔹 Precision: 0.8066
🔹 Recall   : 0.7972


🏆 MKUNet (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.8393 | IoU: 0.7599
🔹 ACC  : 0.9886  | Pre: 0.8699 | Rec: 0.8788
--------------------------------------------------

# 11 HiFormer
HiFormer 

🏆 HiFormer（busi） 最终测试集成绩 🏆
🔹 Dice (F1): 0.7354
🔹 IoU      : 0.6433
🔹 ACC      : 0.9559
🔹 Precision: 0.7475
🔹 Recall   : 0.7650

🏆 HiFormer (UDIAT Finetuned) 测试集最终成绩 🏆
🔹 Dice : 0.8631 | IoU: 0.7696
🔹 ACC  : 0.9876  | Pre: 0.8604 | Rec: 0.8890
"""

result_busi, result_udiat = sort_and_filter_models(raw_md)

# 保存文件
with open('sorted_by_busi_dice.txt', 'w', encoding='utf-8') as f:
    f.write(result_busi)

with open('sorted_by_udiat_dice.txt', 'w', encoding='utf-8') as f:
    f.write(result_udiat)

print("处理完成！生成的文档已剔除无关成绩。")