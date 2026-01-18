import os
import json
import ast
import pandas as pd
from PIL import Image
import io
import base64
from openai import OpenAI

# =========================================================================
# 🛠️ 【全局配置区域】 - 所有路径和设置都在这里修改
# =========================================================================

# 1. 待测评图片的文件夹路径
INPUT_ARTWORK_FOLDER = "images"

# 2. 测评结果保存的 Excel 路径
#    注意：如果这里只写文件名(如 "result.xlsx")，代码已修复，不会再报错了
OUTPUT_EXCEL_PATH = "result.xlsx"

# 3. API 配置文件路径
API_CONFIG_FILE = "config.json"

# 4. Few-shot 参考案例数据文件路径 (JSON格式)
FEW_SHOT_DATA_FILE = "few_shot_data.json"

# 5. 图片压缩设置 (用于控制 Token 消耗)
IMG_MAX_WIDTH = 400
IMG_QUALITY = 25


# =========================================================================
# 🚀 以下是核心逻辑代码
# =========================================================================

def load_json_file(filepath):
    """通用 JSON 读取函数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filepath}，请检查路径。")
        exit()
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {filepath} 格式不正确，请检查 JSON 语法。")
        exit()


# 加载配置
api_config = load_json_file(API_CONFIG_FILE)

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=api_config.get("base_url"),
    api_key=api_config.get("api_key")
)


def encode_image(image_path, max_width=800, quality=50):
    """压缩并编码图片为 Base64"""
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            if img.width > max_width:
                ratio = max_width / float(img.width)
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ 图片处理失败: {image_path} | 错误: {e}")
        return None


def Multi_shot_analyze_image(target_image_base64, few_shot_examples):
    """
    动态 Multi-shot 分析函数
    """
    # 1. 基础系统提示词 (评分标准)
    system_prompt_criteria = """
你是一位资深的艺术教授，拥有卓越的艺术鉴赏天赋，特别擅长评价儿童美术作品。
请使用以下标准评估学生的艺术作品，并对每个标准给出一个分数（1-5分）。

维度 1：写实 (Realistic)。评估比例、纹理、光影和透视的准确性。
5：细节非凡，光影纹理大师级，比例透视精准，极其逼真。
4：细节高水平，比例纹理佳，光影增强真实感，微小瑕疵。
3：中等写实，基本比例正确，有光影纹理，但缺深度或细节。
2：试图写实但比例纹理挣扎，光影透视不一致，缺乏说服力。
1：几乎不写实，比例光影糟糕。

维度 2：变形 (Deformation)。评估是否有意通过变形传达信息或情感。
5：大师级变形，增强情感冲击，与构图无缝结合。
4：有效利用变形表达意图，整合得当。
3：有明显变形，增加表现力，但可能与构图脱节。
2：试图变形但成功有限，感觉强行或肤浅。
1：变形极少或无效，与意图脱节。

维度 3：想象 (Imagination)。评估原创性和创造力。
5：极高原创性，概念独特令人惊讶。
4：创意原创且执行良好，虽可能类同传统。
3：有些创意，但稍显可预测。
2：创意极少，主要模仿。
1：缺乏想象，无原创想法。

维度 4：色彩丰富度 (Color Richness)。评估色彩范围和视觉体验。
5：色彩广泛和谐，生动充满活力。
4：色彩多样平衡，增强吸引力。
3：中等色彩范围，未完全衬托主题。
2：色彩多样性有限。
1：色彩运用糟糕，范围受限。

维度 5：色彩对比度 (Color Contrast)。评估对比色的使用。
5：巧妙运用对比，视觉冲击惊人。
4：有效使用对比，增强趣味。
3：有些对比，但效果一般。
2：极少对比，视觉平淡。
1：缺乏对比，无吸引力。

维度 6：线条组合 (Line Combination)。评估线条整合与交互。
5：线条整合非凡，视觉流动和谐。
4：线条组合良好，部分区域缺凝聚力。
3：线条组合平均，缺乏整体连贯。
2：极少有效使用，线条冲突。
1：线条整合糟糕，破坏和谐。

维度 7：线条纹理 (Line Texture)。评估线条纹理多样性和执行。
5：纹理多样，执行巧夺天工。
4：纹理范围好，部分缺定义。
3：纹理中等，细节不足。
2：纹理有限，贡献不大。
1：缺乏纹理多样性和精致度。

维度 8：图像构成 (Picture Organization)。评估构图和空间安排。
5：组织无懈可击，构图平衡引人注目。
4：组织良好，引导视线，微小干扰。
3：组织尚可，略显不平衡。
2：组织糟糕，缺乏连贯。
1：组织混乱，杂乱无章。

维度 9：转化 (Transformation)。评估将传统元素转化为新颖事物的能力。
5：转化性强，视角新颖创新。
4：成功转化熟悉元素，新视角。
3：有些转化，但可预测。
2：试图转化但成功微小。
1：缺乏转化，单纯复制。
"""

    # 2. 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt_criteria},
        {"role": "system", "content": "请根据以下提供的参考示例（Few-shot），学习其评分尺度，然后为新图像评分。"}
    ]

    # 3. 动态插入 Few-shot 示例
    for idx, example in enumerate(few_shot_examples, 1):
        content_str = f"参考示例 {idx}: \n专家评分: {json.dumps(example['scores'], ensure_ascii=False)} \n评价理由/备注: {example.get('comment', '无')}"

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": content_str},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{example['base64']}", "detail": "low"}
                }
            ]
        })
        messages.append({
            "role": "assistant",
            "content": "收到，我已学习该示例的评分标准。"
        })

    # 4. 插入待测图片任务
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "现在请评估这张新图片。输出格式必须为Python字典: {'写实': 分数, '变形': 分数, '想象': 分数, '色彩丰富度': 分数, '色彩对比度': 分数, '线条组合': 分数, '线条纹理': 分数, '图像构成': 分数, '转化': 分数}。只返回字典，不要Markdown。"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{target_image_base64}", "detail": "high"}
            }
        ]
    })

    # 5. 调用 API
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=messages,
        max_tokens=300,
    )
    return response.choices[0].message.content


def safe_eval_dict(input_string):
    """安全解析字典"""
    cleaned = input_string.replace("```json", "").replace("```python", "").replace("```", "").strip()
    try:
        return ast.literal_eval(cleaned)
    except:
        try:
            fixed = cleaned.replace("{", '{"').replace(", ", ', "').replace(":", '":')
            return ast.literal_eval(fixed)
        except:
            return {}


# =========================================================================
# 🏁 主程序入口 (已修复 Bug)
# =========================================================================

def main():
    # 1. 准备环境
    # 【核心修复】：先判断路径是否为空，防止 "result.xlsx" 这种纯文件名报错
    output_dir = os.path.dirname(OUTPUT_EXCEL_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(INPUT_ARTWORK_FOLDER, exist_ok=True)

    # 2. 加载并预处理 Few-shot 数据
    print(f"📚 正在从 {FEW_SHOT_DATA_FILE} 加载参考案例...")
    try:
        raw_few_shots = load_json_file(FEW_SHOT_DATA_FILE)
    except Exception as e:
        print(f"❌ 无法读取参考案例文件: {e}")
        return

    processed_few_shots = []
    for shot in raw_few_shots:
        path = shot.get("image_path")
        if path and os.path.exists(path):
            b64 = encode_image(path, IMG_MAX_WIDTH, IMG_QUALITY)
            if b64:
                processed_few_shots.append({
                    "base64": b64,
                    "scores": shot.get("scores"),
                    "comment": shot.get("comment", "")
                })
        else:
            print(f"⚠️ 警告: 参考图片不存在，跳过该示例: {path}")

    print(f"✅ 成功加载 {len(processed_few_shots)} 个参考示例 (Few-shot)。")

    # 3. 扫描待测文件夹
    if not os.path.exists(INPUT_ARTWORK_FOLDER):
        print(f"❌ 文件夹 {INPUT_ARTWORK_FOLDER} 不存在。")
        return

    file_list = [f for f in os.listdir(INPUT_ARTWORK_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not file_list:
        print("📭 待测文件夹为空。请在 images 文件夹中放入图片。")
        return

    results = []

    # 定义固定的列顺序，防止分数错位
    metrics_columns = ["写实", "变形", "想象", "色彩丰富度", "色彩对比度", "线条组合", "线条纹理", "图像构成", "转化"]

    # 4. 循环处理图片
    print("-" * 50)
    for idx, filename in enumerate(file_list, 1):
        filepath = os.path.join(INPUT_ARTWORK_FOLDER, filename)
        print(f"[{idx}/{len(file_list)}] 正在评估: {filename}")

        target_b64 = encode_image(filepath, IMG_MAX_WIDTH, IMG_QUALITY)
        if not target_b64:
            continue

        # 调用分析
        try:
            resp_str = Multi_shot_analyze_image(target_b64, processed_few_shots)
            scores = safe_eval_dict(resp_str)

            if scores:
                # 【核心修复】：按照固定顺序取值，防止 AI 返回乱序导致 Excel 列错位
                ordered_scores = [scores.get(col, 0) for col in metrics_columns]

                row = [idx, filename] + ordered_scores
                results.append(row)
                print(f"   ---> 评估完成。")
            else:
                print(f"   ---> ❌ 格式解析失败 (AI 返回数据异常)。")

        except Exception as e:
            print(f"   ---> ❌ API 请求出错: {e}")

    # 5. 保存结果
    if results:
        columns = ["ID", "Filename"] + metrics_columns
        df = pd.DataFrame(results, columns=columns)
        try:
            df.to_excel(OUTPUT_EXCEL_PATH, index=False)
            print("-" * 50)
            print(f"💾 任务完成！结果已保存至: {OUTPUT_EXCEL_PATH}")
        except PermissionError:
            print(f"❌ 保存失败: 文件被占用，请先关闭 {OUTPUT_EXCEL_PATH}！")
    else:
        print("没有生成任何有效数据。")


if __name__ == "__main__":
    main()