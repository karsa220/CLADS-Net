import os
import sys
import requests
import warnings
import importlib.util
import numpy as np
import matplotlib.pyplot as plt  # 🟢 新增导入 matplotlib
from transformers import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ==========================================
# 🌟 飞书机器人推送函数
# ==========================================
def send_feishu_message(model_name, test_res):
    webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/a3786d6b-5e74-4b23-b857-36e19a3a9c62"

    try:
        if isinstance(test_res, dict):
            dice = np.mean(test_res.get('dice', 0))
            iou = np.mean(test_res.get('iou', 0))
            acc = np.mean(test_res.get('acc', 0))
            pre = np.mean(test_res.get('pre', 0))
            rec = np.mean(test_res.get('rec', 0))
        elif isinstance(test_res, (list, tuple)):
            dice, iou, acc, pre, rec = test_res[0], test_res[1], test_res[2], test_res[3], test_res[4]
        else:
            print(f"⚠️ 模型 [{model_name}] 返回的数据格式不支持！")
            return
    except Exception as e:
        print(f"⚠️ 解析模型 [{model_name}] 结果时出错: {e}")
        return

    text = (f"🎉 报告老板，模型 [{model_name}] 训练及测试完毕！\n\n"
            f"🏆 最终测试集成绩:\n"
            f"🔹 Dice     : {dice:.4f}\n"
            f"🔹 IoU      : {iou:.4f}\n"
            f"🔹 ACC      : {acc:.4f}\n"
            f"🔹 Precision: {pre:.4f}\n"
            f"🔹 Recall   : {rec:.4f}\n")

    msg = {"msg_type": "text", "content": {"text": text}}
    try:
        requests.post(webhook_url, json=msg)
    except Exception as e:
        pass


# ==========================================
# 🌟 动态加载主函数 (核心隔离区)
# ==========================================
def main():
    master_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    all_items = os.listdir(master_dir)

    model_folders = [item for item in all_items
                     if os.path.isdir(os.path.join(master_dir, item))
                     and os.path.isfile(os.path.join(master_dir, item, "main.py"))]

    print(f"🔍 找到以下包含 main.py 的模型文件夹：{model_folders}\n")

    for folder_name in model_folders:
        folder_path = os.path.join(master_dir, folder_name)
        main_script_path = os.path.join(folder_path, "main.py")

        print(f"\n{'=' * 50}\n🚀 正在加载并执行模型: {folder_name} ...")

        loaded_modules_before = set(sys.modules.keys())

        os.chdir(folder_path)
        sys.path.insert(0, folder_path)

        try:
            spec = importlib.util.spec_from_file_location(f"{folder_name}_main", main_script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'main'):
                test_res = module.main()

                if test_res is not None:
                    send_feishu_message(folder_name, test_res)
                    print(f"✅ 模型 [{folder_name}] 执行完毕并已推送到飞书。")
                else:
                    print(f"⚠️ 模型 [{folder_name}] 的 main() 返回了 None。")
            else:
                print(f"⚠️ 文件夹 {folder_name} 下没有找到 main() 函数！")

        except Exception as e:
            print(f"❌ 运行模型 [{folder_name}] 时发生错误: {e}")

        finally:
            # 1. 恢复系统路径和工作目录
            if folder_path in sys.path:
                sys.path.remove(folder_path)
            os.chdir(master_dir)

            # 2. 🟢 强制关闭并清理本轮产生的所有 Matplotlib 绘图，释放内存
            plt.close('all')

            # 3. 清理该模型加载的本地专属模块
            modules_to_remove = []
            for mod_name in list(sys.modules.keys()):
                if mod_name not in loaded_modules_before:
                    mod = sys.modules.get(mod_name)
                    if mod and hasattr(mod, '__file__') and mod.__file__ and mod.__file__.startswith(folder_path):
                        modules_to_remove.append(mod_name)

            for mod_name in modules_to_remove:
                del sys.modules[mod_name]

    print("\n🎉🎉🎉 所有对比模型训练及测试流程已全部结束！")


if __name__ == "__main__":
    main()