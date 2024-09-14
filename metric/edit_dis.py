
import concurrent.futures
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
def min_edit_distance_for_gen_seq(gen_seq, reference_sequences):
    min_edit_distance = float('inf')
    for ref_seq in reference_sequences:
        edit_distance = levenshtein_distance(gen_seq, ref_seq)
        if edit_distance < min_edit_distance:
            min_edit_distance = edit_distance
    return min_edit_distance


edit_distances = []
def read_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file.readlines()]
    return sequences

generated_sequences_file = "/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/result/finetune_generated_sequences.txt"
reference_sequences_file = "/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/Data/AMP/AMP.csv"

generated_sequences = read_sequences(generated_sequences_file)
reference_sequences = read_sequences(reference_sequences_file)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    future_min_edit_distances = {executor.submit(min_edit_distance_for_gen_seq, gen_seq, reference_sequences): gen_seq for gen_seq in generated_sequences}

    for future in concurrent.futures.as_completed(future_min_edit_distances):
        edit_distances.append(future.result())




import matplotlib.pyplot as plt

plt.hist(edit_distances, bins='auto', edgecolor='black')
plt.xlabel('min edit distance')
plt.ylabel('num')
plt.title('edit_distance_distribution')

plt.savefig('edit_distance_distribution.png')