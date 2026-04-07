import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib

# 指定字体为 SimHei（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
# 1. 读取两个Excel文件
file1_path = r""
file2_path = r""

# 读取第一个文件的Sheet1
df1 = pd.read_excel(file1_path, sheet_name="Sheet1")

# 读取第二个文件的Sheet1
df2 = pd.read_excel(file2_path, sheet_name="Sheet1")

# 2. 提取需要对比的评分列（排除非评分列）
score_columns = ["写实", "变形", "想象", "色彩丰富度", "色彩对比度", "线条组合", "线条纹理", "图像构成", "转化"]

# 检查列名是否一致
if list(df1[score_columns].columns) != list(df2[score_columns].columns):
    raise ValueError("两个文件的评分列名称不一致，请检查数据格式")

# 3. 按行顺序对齐数据（假设行顺序一致）
scores1 = df1[score_columns].values
scores2 = df2[score_columns].values

if len(scores1) != len(scores2):
    raise ValueError("两个文件的数据行数不一致，无法直接对比")

# 4. 计算斯皮尔曼相关系数
results = []
for i, col in enumerate(score_columns):
    corr, p_value = spearmanr(scores1[:, i], scores2[:, i])
    results.append({
        "维度": col,
        "斯皮尔曼相关系数": round(corr, 3),
        "P值": round(p_value, 5)
    })

# 5. 输出结果
result_df = pd.DataFrame(results)
print("\n斯皮尔曼相关系数结果：")
print(result_df.to_string(index=False))
