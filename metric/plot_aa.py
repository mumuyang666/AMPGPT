import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
natural_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/max_predictions.txt'
artificial_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/MIC/min_predictions.txt'
# 提取氨基酸序列
def extract_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            seq = line.strip()
            sequences.append(seq)
    return sequences
# 计算氨基酸频率
def count_amino_acids(sequences):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts = {aa: 0 for aa in amino_acids}
    total = 0

    for seq in sequences:
        for aa in seq:
            if aa in counts:
                counts[aa] += 1
                total += 1

    frequencies = {aa: count / total for aa, count in counts.items()}
    return frequencies

# 提取每个文件中的氨基酸序列
file1_sequences = extract_sequences(natural_path)
file2_sequences = extract_sequences(artificial_path)


# 计算每个文件中的氨基酸频率
file1_frequencies = count_amino_acids(file1_sequences)
file2_frequencies = count_amino_acids(file2_sequences)
p_values = {}
for aa in file1_frequencies.keys():
    count = [file1_frequencies[aa] * len(file1_sequences), file2_frequencies[aa] * len(file2_sequences)]
    nobs = [len(file1_sequences), len(file2_sequences)]
    z_stat, p_value = proportions_ztest(count, nobs)
    p_values[aa] = p_value

print("P values:")
for aa, p_value in p_values.items():
    print(f"{aa}: {p_value}")
print("natural\n")
print(file1_frequencies)
print("artificial\n")
print(file2_frequencies)
fig, ax = plt.subplots(figsize=(8,6))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15
# 绘制柱状图

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
# 将数据整理成适合绘图的格式
data = {
    'Amino Acid': list(file1_frequencies.keys()),
    'GPT_MIChigh_AMP': list(map(float, file1_frequencies.values())),
    'GPT_MIClow_AMP': list(map(float, file2_frequencies.values()))
}
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, aa in enumerate(data_df['Amino Acid']):
    plt.text(i, max(data_df['GPT_MIChigh_AMP'][i], data_df['GPT_MIClow_AMP'][i]), f"p = {p_values[aa]:.2e}", ha='center')
# 转换为Pandas DataFrame格式
data_df = pd.DataFrame(data)
color = sns.color_palette("Set2")# 绘制堆叠柱状图
color = ["#D6AFB9", "#7E9BB7"]
ax = data_df.plot.bar(x='Amino Acid', stacked=False, figsize=(10, 4), color=color, edgecolor='black')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels(data_df['Amino Acid'], rotation=0)
# plt.grid()
plt.tight_layout()
#ax.set_title('Amino Acid Distribution by JSONL File')
plt.savefig('MIC_min_distribution_source.png', dpi=300)
# plt.savefig('fintune2_distribution_source.pdf', dpi=300)