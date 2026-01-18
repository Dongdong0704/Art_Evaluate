import os
import json
import time
import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm  # 进度条库
from cozepy import Coze, TokenAuth, Message, ChatEventType, COZE_CN_BASE_URL

# ---- 配置 ----
# 日志级别设为 WARNING，让进度条显示更干净
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# 【重要】根据您提供的 JSON 样例，更新为这 9 个维度
TARGET_DIMENSIONS = [
    "写实",
    "变形",
    "想象",
    "色彩丰富度",
    "色彩对比度",
    "线条组合",
    "线条纹理",
    "图像构成",
    "转化"
]


# ---- 辅助函数 ----
def load_config(path='config.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_json_string(text):
    """
    清洗数据，从 Markdown 代码块中提取 JSON
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 匹配 ```json { ... } ``` 或 { ... }
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return None


# ---- 核心处理逻辑 ----
def process_single_image(coze_client, bot_id, user_id, file_path):
    """
    处理单张图片：上传 -> 对话 -> 解析您特定的嵌套 JSON 格式
    """
    filename = os.path.basename(file_path)
    result_row = {"图片文件名": filename}

    try:
        # 1. 上传
        file_obj = coze_client.files.upload(file=Path(file_path))
        file_id = file_obj.id

        # 2. 构造 Prompt
        # 虽然您的 bot 已经会输出 JSON，但我们在 Prompt 里再强调一遍格式，双重保险
        prompt = (
            f"请对这张作品进行评分。请严格返回JSON格式数据。"
            f"你需要评估的维度包括：{json.dumps(TARGET_DIMENSIONS, ensure_ascii=False)}。"
            f"返回格式必须符合：{{ \"写实\": {{ \"score\": 1 }}, ... }}，不要包含其他废话。"
        )

        content_payload = [
            {"type": "text", "text": prompt},
            {"type": "image", "file_id": file_id}
        ]

        messages = [
            Message(
                role="user",
                content_type="object_string",
                content=json.dumps(content_payload, ensure_ascii=False)
            )
        ]

        # 3. 请求
        full_response = ""
        for event in coze_client.chat.stream(
                bot_id=bot_id,
                user_id=user_id,
                additional_messages=messages,
        ):
            if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                full_response += event.message.content

            if event.event == ChatEventType.CONVERSATION_CHAT_COMPLETED:
                if event.chat.status == "failed":
                    return {**result_row, "状态": f"API错误: {event.chat.last_error.msg}"}

        # 4. 解析逻辑 (针对您的特定 JSON 结构)
        parsed_data = clean_json_string(full_response)

        if parsed_data and isinstance(parsed_data, dict):
            # 遍历9个目标维度，精准提取 score
            for dim in TARGET_DIMENSIONS:
                dim_data = parsed_data.get(dim)

                # 情况A: 数据是 {"写实": {"score": 5}} (您的标准格式)
                if isinstance(dim_data, dict) and "score" in dim_data:
                    result_row[dim] = dim_data["score"]

                # 情况B: 数据偶尔变成 {"写实": 5} (容错处理)
                elif isinstance(dim_data, (int, float, str)):
                    result_row[dim] = dim_data

                # 情况C: 没找到这个维度
                else:
                    result_row[dim] = None

            result_row["状态"] = "成功"
        else:
            result_row["状态"] = "解析失败"
            # 如果解析失败，把原始回复存到第一列的备注里，方便排查
            result_row["原始回复"] = full_response[:500]

    except Exception as e:
        result_row["状态"] = f"程序异常: {str(e)}"

    return result_row


# ---- 批量并发调度 ----
def process_folder_concurrently(config):
    coze_api_token = config.get('coze_api_token')
    bot_id = config.get('bot_id')
    user_id = config.get('user_id')

    # 线程数：3-5 是比较安全的范围
    max_workers = config.get('max_workers', 3)

    if not coze_api_token or not bot_id:
        print("错误：Config配置缺失")
        return

    coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=COZE_CN_BASE_URL)

    input_folder = config.get('input_folder', './images')
    output_folder = config.get('output_folder', './results')
    excel_path = config.get('excel_path', os.path.join(output_folder, '美术测评结果.xlsx'))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    if not os.path.exists(input_folder):
        print(f"文件夹不存在: {input_folder}")
        return

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if Path(f).suffix.lower() in supported]
    total_files = len(files)

    if total_files == 0:
        print("文件夹为空。")
        return

    print(f"🚀 开始处理 {total_files} 张图片，并发线程数: {max_workers}")
    print(f"📋 提取维度: {TARGET_DIMENSIONS}")

    all_results = []

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, coze, bot_id, user_id, f) for f in files]

        # 进度条显示
        for future in tqdm(as_completed(futures), total=total_files, unit="img", desc="测评进度"):
            result = future.result()
            all_results.append(result)

    print("\n💾 正在保存 Excel...")
    df = pd.DataFrame(all_results)

    # 整理列顺序：文件名 -> 状态 -> 9个维度
    cols = ["图片文件名", "状态"] + TARGET_DIMENSIONS

    # 确保Excel里只保留这些列，防止其他杂乱字段（如果有的话）
    final_cols = [c for c in cols if c in df.columns]
    # 如果有“原始回复”列（解析失败时），也加上
    if "原始回复" in df.columns:
        final_cols.append("原始回复")

    df = df[final_cols]

    try:
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"✅ 完成！表格已生成: {excel_path}")
    except Exception as e:
        print(f"❌ 保存失败 (Excel可能被占用): {e}")
        df.to_csv(os.path.join(output_folder, "backup.csv"), index=False)


if __name__ == "__main__":
    try:
        config = load_config()
        process_folder_concurrently(config)
    except Exception as e:
        print(f"主程序错误: {e}")