import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

# ======================================================
# 👉 第一步：请在此处填入您的本地文件路径
# ======================================================

# --- Excel文件 1 (例如：人工/专家评分) ---
FILE_PATH_1 = r""

# --- Excel文件 2 (例如：模型/AI评分) ---
FILE_PATH_2 = r""


# ======================================================
# 第二步：核心计算逻辑 (自动批量处理)
# ======================================================

def batch_calculate_qwk():
    print("-" * 50)
    print("🚀 开始批量计算 QWK...")

    # 1. 读取两个 Excel 文件
    try:
        df1 = pd.read_excel(FILE_PATH_1)
        df2 = pd.read_excel(FILE_PATH_2)
        print(f"✅ 成功读取文件 1: {os.path.basename(FILE_PATH_1)} (共 {len(df1)} 行)")
        print(f"✅ 成功读取文件 2: {os.path.basename(FILE_PATH_2)} (共 {len(df2)} 行)")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 2. 找出两个表格中【共有的列名】
    # 只有两个表都有的列，才能进行对比计算
    common_columns = [col for col in df1.columns if col in df2.columns]

    # 过滤掉可能存在的非评分列（比如 '姓名', '学号', 'ID' 等，如果它们也是数字可能被误算）
    # 这里默认计算所有同名列，如果您有不需要算的列，可以在结果里忽略
    if not common_columns:
        print("❌ 错误：两个 Excel 表格没有发现相同的列名，无法对比。请检查表头是否一致。")
        return

    print(f"🔍 发现 {len(common_columns)} 个共同评价维度: {common_columns}")
    print("-" * 50)

    # 3. 对齐行数 (取最小行数，防止行数不一致报错)
    min_len = min(len(df1), len(df2))
    if len(df1) != len(df2):
        print(f"⚠️ 注意：行数不一致，将只计算前 {min_len} 行的数据。")

    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

    # 4. 循环计算每一列的 QWK
    results = []

    print(f"{'维度名称':<15} | {'QWK 分数':<10} | {'一致性评价'}")
    print("-" * 50)

    for col in common_columns:
        try:
            # 获取这两列数据
            y_true = df1[col]  # 人工
            y_pred = df2[col]  # 模型

            # 确保数据是数值型（排除纯文本列）
            if not pd.api.types.is_numeric_dtype(y_true) or not pd.api.types.is_numeric_dtype(y_pred):
                # 如果不是数字，跳过
                continue

            # 计算二次加权 Kappa
            score = cohen_kappa_score(y_true, y_pred, weights='quadratic')

            # 简单的评价描述
            comment = ""
            if score <= 0.2:
                comment = "极低"
            elif score <= 0.4:
                comment = "一般"
            elif score <= 0.6:
                comment = "中等"
            elif score <= 0.8:
                comment = "高度"
            else:
                comment = "极高"

            # 打印到控制台
            print(f"{col:<15} | {score:.4f}     | {comment}")

            # 保存结果到列表
            results.append({
                '评价维度': col,
                'QWK系数': score,
                '一致性水平': comment
            })

        except Exception as e:
            print(f"⚠️ 无法计算列 '{col}': {e}")

    # 5. 将结果导出到一个新的 Excel 文件，方便您复制使用
    if results:
        result_df = pd.DataFrame(results)
        output_filename = "QWK_计算结果报告.xlsx"
        result_df.to_excel(output_filename, index=False)
        print("-" * 50)
        print(f"💾 所有维度的计算结果已保存至: {output_filename}")
        print("您可以直接打开这个文件查看所有列的对比结果。")
    else:
        print("没有计算出任何有效结果，请检查数据列是否包含数值。")


if __name__ == "__main__":
    batch_calculate_qwk()