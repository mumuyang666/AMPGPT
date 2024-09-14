# 创建模型实例
import heapq
from GPT import GPT_Model
import torch
import random
import torch.nn as nn
import torch.utils.data as Data
amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYWOBXJZ"]
token_list = amino_acids  # 只使用氨基酸作为tokens
token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[POS]': 3, '[NEG]': 4 }
for i, w in enumerate(token_list):
    token2idx[w] = i + 5
idx2token = {i: w for i, w in enumerate(token2idx)}
vocab_size = len(token2idx)


def generate_sequence(model, start_token, max_len):
    model.eval()
    input_ids = [start_token]
    device = next(model.parameters()).device  # 获取模型所在的设备
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)  # 将输入数据移动到模型所在的设备
            _, logits = model(input_tensor)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1)
            input_ids.append(predicted_id.item())
    return input_ids
def generate_sequence_topk(model, start_token, max_len, top_k=10):
    model.eval()
    input_ids = [start_token]
    device = next(model.parameters()).device  # 获取模型所在的设备
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)  # 将输入数据移动到模型所在的设备
            _, logits = model(input_tensor)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # 使用topk来获取最高概率的token
            top_k_probs, top_k_ids = torch.topk(probs, k=top_k, dim=-1)
            # 从top_k_ids中随机选择一个作为预测的id
            choices = torch.multinomial(top_k_probs, num_samples=1)
            predicted_id = top_k_ids[0, choices[0]].item()
            input_ids.append(predicted_id)
    return input_ids

def generate_sequences(model, start_token, max_len, num_sequences, top_k=10):
    sequences = []
    for _ in range(num_sequences):
        sequence = generate_sequence_topk(model, start_token, max_len, top_k)
        sequences.append(sequence)
    return sequences
# 使用模型和开始符号生成一个序列
def load_model(model_weights_path, vocab_size, d_model):
    model = GPT_Model(vocab_size, d_model)
    model.load_state_dict(torch.load(model_weights_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model

seed_n = 0
random.seed(seed_n)
torch.manual_seed(seed_n)
# 设置参数
model_weights_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/model/gpt_finetune_best_val_epoch_6.pt'
vocab_size = len(token2idx)
d_model = 768  # 可根据实际情况修改
start_token = token2idx['[NEG]']
max_len = 52
# 加载模型
model = load_model(model_weights_path, vocab_size, d_model)
generated_sequences = generate_sequences(model, start_token, max_len, num_sequences=100, top_k=10)

# 将生成的序列转换为token表示
generated_sequences_tokens = []
for seq in generated_sequences:
    seq_tokens = [idx2token[idx] for idx in seq]
    try:
        sep_index = seq_tokens.index('[SEP]')
        seq_tokens = seq_tokens[1:sep_index]
    except ValueError:
        seq_tokens = seq_tokens[1:]
    generated_sequences_tokens.append(seq_tokens)

# 打印生成的序列
for i, seq in enumerate(generated_sequences_tokens):
    print(f"Generated sequence {i + 1}: {''.join(seq)}")

# 将生成的序列写入 txt 文件
with open("./result/fintune_generated_sequences_neg.txt", "w") as f:
    for seq in generated_sequences_tokens:
        f.write("".join(seq) + "\n")
# 使用模型和开始符号生成一个序列
# 使用模型和开始符号生成一组序列
# top_generated_sequences = generate_sequence_beam_search(model, start_token=token2idx['[CLS]'], max_len=52, beam_size=10)

# 将生成的序列转换为 tokens
# for i, seq in enumerate(top_generated_sequences):
#     generated_tokens = [idx2token[idx] for idx in seq]
#     print(f"Top {i+1} sequence: {generated_tokens}")
# def top_k_sampling(logits, k):
#     # 将 logits 转换为概率分布
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     # 获取 top-k 的概率值和索引
#     top_k_probs, top_k_indices = torch.topk(probs, k)
#     # 对 top-k 的概率进行重新归一化
#     top_k_probs = top_k_probs / top_k_probs.sum()
#     # 对 top-k 的概率进行采样
#     sampled_idx = torch.multinomial(top_k_probs, 1)
#     # 返回采样得到的索引
#     return top_k_indices[0, sampled_idx].item()

# def generate_sequence_beam_search(model, start_token, max_len, beam_size, top_k=10):
#     model.eval()
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         beams = [(0, [start_token])]
#         for _ in range(max_len):
#             new_beams = []
#             for score, beam in beams:
#                 input_tensor = torch.tensor(beam).unsqueeze(0).to(device)
#                 _, logits = model(input_tensor)
#                 logits = logits[:, -1, :]
#                 probs = torch.nn.functional.softmax(logits, dim=-1)
#                 top_probs, top_indices = torch.topk(probs, beam_size)
#                 for i in range(beam_size):
#                     new_score = score + top_probs[0][i].item()
#                     new_beam = beam + [top_indices[0][i].item()]
#                     new_beams.append((new_score, new_beam))
#             beams = heapq.nlargest(beam_size, new_beams, key=lambda x: x[0])
#     top_beams = heapq.nlargest(top_k, beams, key=lambda x: x[0])
#     top_sequences = [beam[1] for beam in top_beams]
#     return top_sequences
