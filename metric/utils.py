import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import math
# def extract_sequences(file_path):
#     sequences = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             if 'trg' in data:
#                 seq = data['trg']
#             elif 'recover' in data:
#                 seq = data['recover']
#             sequences.append(seq)
#     return sequences

def compute_aromaticity(seq):
    analysis = ProteinAnalysis(seq)
    return analysis.aromaticity()

# 氨基酸电荷字典
charge_dict = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# 计算序列的电荷
def compute_charge(seq, charge_dict=charge_dict):
    total_charge = 0
    count = 0
    for aa in seq:
        if aa in charge_dict:
            total_charge += charge_dict[aa]
            count += 1
    return total_charge / count if count > 0 else None

# Kyte-Doolittle疏水性指数
hydrophobicity_dict = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}
# 计算序列的全局疏水性
def compute_global_hydrophobicity(seq, hydrophobicity_dict=hydrophobicity_dict):
    total_hydrophobicity = 0
    count = 0
    for aa in seq:
        if aa in hydrophobicity_dict:
            total_hydrophobicity += hydrophobicity_dict[aa]
            count += 1
    return total_hydrophobicity / count if count > 0 else None


# Eisenberg 疏水性指数
hydrophobicity_dict = {
    'A':  0.62, 'C':  0.29, 'D': -0.90, 'E': -0.74, 'F':  1.19,
    'G':  0.48, 'H': -0.40, 'I':  1.38, 'K': -1.50, 'L':  1.06,
    'M':  0.64, 'N': -0.78, 'P':  0.12, 'Q': -0.85, 'R': -2.53,
    'S': -0.18, 'T': -0.05, 'V':  1.08, 'W':  0.81, 'Y':  0.26
}
# 计算序列的全局疏水矩
def compute_global_hydrophobic_moment(seq, hydrophobicity_dict=hydrophobicity_dict):
    angle_rad = 100 * (math.pi / 180)  # 100 degrees in radians
    moment = 0
    count = 0
    for i, aa in enumerate(seq):
        if aa in hydrophobicity_dict:
            moment += hydrophobicity_dict[aa] * math.cos((i % 3) * angle_rad)
            count += 1
    return moment / count if count > 0 else None

# 计算序列的等电点
def compute_isoelectric_point(seq):
    analysis = ProteinAnalysis(seq)
    return analysis.isoelectric_point()
import csv
# 从文件中提取 trg 或 recover 的长度
def compute_lengths(file_path):
    lengths = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            length = len(row[0])
            if length <= 50:
                lengths.append(length)
    return lengths