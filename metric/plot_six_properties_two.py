from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import scipy.stats as stats
def extract_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            seq = row[0].strip()  # 获取第一列的数据
            sequences.append(seq)
    return sequences
natural_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/result/fintune_generated_sequences_neg.txt'
artificial_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/result/finetune_generated_sequences.txt'
natural_sequences = extract_sequences(natural_path)
# print(natural_sequences.shape)
artificial_sequences = extract_sequences(artificial_path)

# Calculate aromaticity
natural_aromaticity = [compute_aromaticity(seq) for seq in natural_sequences]
artificial_aromaticity = [compute_aromaticity(seq) for seq in artificial_sequences]

# Calculate charge for each sequence in the files
natural_charge = [compute_charge(seq, charge_dict) for seq in natural_sequences]
artificial_charge = [compute_charge(seq, charge_dict) for seq in artificial_sequences]
# Filter out sequences containing uncommon amino acids
natural_charge = [x for x in natural_charge if x is not None]
artificial_charge = [x for x in artificial_charge if x is not None]

# Calculate global hydrophobicity for each sequence in the files
natural_hydrophobicity = [compute_global_hydrophobicity(seq, hydrophobicity_dict) for seq in natural_sequences]
artificial_hydrophobicity = [compute_global_hydrophobicity(seq, hydrophobicity_dict) for seq in artificial_sequences]

# Calculate global hydrophobic moment for each sequence in the files
natural_hydrophobic_moment = [compute_global_hydrophobic_moment(seq, hydrophobicity_dict) for seq in natural_sequences]
artificial_hydrophobic_moment = [compute_global_hydrophobic_moment(seq, hydrophobicity_dict) for seq in artificial_sequences]
# Filter out sequences containing uncommon amino acids
natural_hydrophobic_moment = [x for x in natural_hydrophobic_moment if x is not None]
artificial_hydrophobic_moment = [x for x in artificial_hydrophobic_moment if x is not None]

# Calculate isoelectric points for each sequence in the files
natural_isoelectric_points = [compute_isoelectric_point(seq) for seq in natural_sequences]
artificial_isoelectric_points = [compute_isoelectric_point(seq) for seq in artificial_sequences]

# Extract lengths of trg or recover in each file
natural_lengths = compute_lengths(natural_path)
artificial_lengths = compute_lengths(artificial_path)

natural_aromaticity = [x for x in natural_aromaticity if x is not None]
artificial_aromaticity = [x for x in artificial_aromaticity if x is not None]

natural_charge = [x for x in natural_charge if x is not None]
artificial_charge = [x for x in artificial_charge if x is not None]

natural_hydrophobicity = [x for x in natural_hydrophobicity if x is not None]
artificial_hydrophobicity = [x for x in artificial_hydrophobicity if x is not None]

natural_hydrophobic_moment = [x for x in natural_hydrophobic_moment if x is not None]
artificial_hydrophobic_moment = [x for x in artificial_hydrophobic_moment if x is not None]

natural_isoelectric_points = [x for x in natural_isoelectric_points if x is not None]
artificial_isoelectric_points = [x for x in artificial_isoelectric_points if x is not None]

natural_lengths = [x for x in natural_lengths if x is not None]
artificial_lengths = [x for x in artificial_lengths if x is not None]
# Create separate DataFrames for each property
def create_df(property_name, natural_values, artificial_values):
    natural_df = pd.DataFrame({property_name: natural_values, 'Type': 'GPT_neg_AMP'})
    artificial_df = pd.DataFrame({property_name: artificial_values, 'Type': 'GPT_pos_AMP'})
    combined_df = pd.concat([natural_df, artificial_df], ignore_index=True)
    return combined_df

aromaticity_df = create_df('Aromaticity', natural_aromaticity, artificial_aromaticity)
charge_df = create_df('Charge', natural_charge, artificial_charge)
hydrophobicity_df = create_df('Hydrophobicity', natural_hydrophobicity, artificial_hydrophobicity)
hydrophobic_moment_df = create_df('Hydrophobic Moment', natural_hydrophobic_moment, artificial_hydrophobic_moment)
isoelectric_point_df = create_df('Isoelectric Point', natural_isoelectric_points, artificial_isoelectric_points)
length_df = create_df('Length', natural_lengths, artificial_lengths)

# Concatenate the DataFrames vertically
combined_df = pd.concat([aromaticity_df, charge_df, hydrophobicity_df, hydrophobic_moment_df, isoelectric_point_df, length_df], axis=0)

# Melt the combined DataFrame
melted_df = combined_df.melt(id_vars=['Type'], var_name='Property', value_name='Value')
# Create the plot
plt.figure(figsize=(16, 8))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 25

properties = ['Aromaticity', 'Charge', 'Hydrophobicity', 'Hydrophobic Moment',  'Length']

palette = ["#D6AFB9", "#7E9BB7"]
for i, prop in enumerate(properties):
    plt.subplot(1, 6, i + 1)
    sns.violinplot(x='Property', y='Value', hue='Type', data=melted_df[melted_df['Property'] == prop],
                   palette=palette, split=False, bw=0.2)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.grid()
    plt.gca().yaxis.tick_left()
    plt.gca().yaxis.set_label_position('left')

    if i > 0:
        plt.gca().yaxis.set_ticks_position('none')

    # Calculate p value
    natural_values = melted_df.loc[(melted_df['Property'] == prop) & (melted_df['Type'] == 'GPT_neg_AMP'), 'Value'].dropna()
    artificial_values = melted_df.loc[(melted_df['Property'] == prop) & (melted_df['Type'] == 'GPT_pos_AMP'), 'Value'].dropna()
    t_stat, p_value = stats.ttest_ind(natural_values, artificial_values)

    # Display p value
    plt.text(0.5, 1.1, f"p = {p_value:.2e}", fontsize=12, ha='center', transform=plt.gca().transAxes)
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = ""
    plt.text(0.5, 1.12, stars, fontsize=16, ha='center', transform=plt.gca().transAxes)
    plt.legend(title=prop, frameon=False, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.1), title_fontsize=18)

plt.tight_layout()
plt.subplots_adjust(wspace=1)
plt.savefig("pos_neg_properties_with_p_values.png", dpi=300)