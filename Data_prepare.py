import torch
import random
from random import *
import torch.utils.data as Data
import pandas as pd
import numpy as np
from utility import shuffle_and_split_data

class Dataset_for_epitope(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
  
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]

class Dataset_for_beta(Data.Dataset):
    def __init__(self, input_ids, output_labels, mask_for_loss):
        self.input_ids = input_ids
        self.output_labels = output_labels
        self.mask_for_loss = mask_for_loss
  
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_labels[idx], self.mask_for_loss[idx]
class Dataset_for_fintune(Data.Dataset):
    def __init__(self, input_ids, output_labels, mask_for_loss, labels):
        self.input_ids = input_ids
        self.output_labels = output_labels
        self.mask_for_loss = mask_for_loss
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_labels[idx], self.mask_for_loss[idx], self.labels[idx]
class Dataset_for_seq2seq(Data.Dataset):
    def __init__(self, dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos):
        self.dec_input_ids = dec_input_ids
        self.output_labels = output_labels
        self.mask_for_loss = mask_for_loss
        self.enc_input_ids = enc_input_ids
        self.masked_pos = masked_pos

    def __len__(self):
        return len(self.dec_input_ids)
    
    def __getitem__(self, idx):
        return self.dec_input_ids[idx], self.output_labels[idx], self.mask_for_loss[idx], self.enc_input_ids[idx], self.masked_pos[idx]

class Dataset_for_generate(Data.Dataset):
    def __init__(self, input, mask):
        self.input = input
        self.mask = mask

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        return self.input[index], self.mask[index]


def make_data_for_pretrain(data_path, vocab_path, maxlen, max_pred):
    data_df = pd.read_csv(data_path)
    seq_list = [seq for seq in data_df['epitope']]
    token_df = pd.read_csv(vocab_path)
    token_list = [token for token in token_df['token']]
    vocab_freq_dict = {token:frequency for token, frequency in zip(token_df['token'],token_df['frequency']) }
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYW"]
    token_list = token_list+amino_acids
    token2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
    for i, w in enumerate(token_list):
        token2idx[w] = i + 4
    idx2token = {i: w for i, w in enumerate(token2idx)}
    vocab_size = len(token2idx)
    token_idx_list = []
    for seq in seq_list:
        tokens = []
        i = 0
        if len(seq)>30:
            seq = seq[:30]
        seq_len = len(seq)
        while i < seq_len:
            if i > seq_len - 2:
                tokens.append(token2idx[seq[i]])
                i += 1
            else:
                temp_token_list = list(set([seq[i: i+token_len] for token_len in [2,3]]))
                temp_token_freq = [vocab_freq_dict[token] if token in vocab_freq_dict.keys() else 0 for token in temp_token_list]
                if sum(temp_token_freq) == 0:
                    tokens.append(token2idx[seq[i]])
                    i += 1
                else:
                    selected_token = temp_token_list[np.argmax(temp_token_freq)]
                    tokens.append(token2idx[selected_token])
                    i += len(selected_token)
        token_idx_list.append(tokens)
    
    batch = []
    for seq in token_idx_list:
        input_ids = [token2idx['[CLS]']] + seq + [token2idx['[SEP]']]
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != token2idx['[CLS]'] and token != token2idx['[SEP]']] # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = token2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        print("Input shape:", input_ids.shape)
        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        batch.append([input_ids, masked_tokens, masked_pos])
    
    train_batch, validate_batch = shuffle_and_split_data(batch, ratio=0.01)
    input_ids, masked_tokens, masked_pos = zip(*train_batch)
    input_ids, masked_tokens, masked_pos = \
    torch.LongTensor(input_ids), torch.LongTensor(masked_tokens),\
    torch.LongTensor(masked_pos)
    input_ids_v, masked_tokens_v, masked_pos_v = zip(*validate_batch)
    input_ids_v, masked_tokens_v, masked_pos_v = \
    torch.LongTensor(input_ids_v), torch.LongTensor(masked_tokens_v),\
    torch.LongTensor(masked_pos_v)
    
    train_dataset = Dataset_for_epitope(input_ids, masked_tokens, masked_pos)
    validate_dataset = Dataset_for_epitope(input_ids_v, masked_tokens_v, masked_pos_v)


    return train_dataset, validate_dataset, vocab_size

# def make_data_for_gpt_pretrain(data_path, vocab_path, max_len):
#     data_df = pd.read_csv(data_path)
#     seq_list = [seq for seq in data_df['beta']]
#     token_df = pd.read_csv(vocab_path)
#     token_list = [token for token in token_df['token']]
#     vocab_freq_dict = {token:frequency for token, frequency in zip(token_df['token'],token_df['frequency']) }
#     amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYW"]
#     token_list = token_list+amino_acids
#     token2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2}
#     for i, w in enumerate(token_list):
#         token2idx[w] = i + 3
#     idx2token = {i: w for i, w in enumerate(token2idx)}
#     vocab_size = len(token2idx)
#     token_idx_list = []
#     for seq in seq_list:
#         tokens = []
#         i = 0
#         if len(seq)>30:
#             seq = seq[:30]
#         seq_len = len(seq)
#         while i < seq_len:
#             if i > seq_len - 2:
#                 tokens.append(token2idx[seq[i]])
#                 i += 1
#             else:
#                 temp_token_list = list(set([seq[i: i+token_len] for token_len in [2,3]]))
#                 temp_token_freq = [vocab_freq_dict[token] if token in vocab_freq_dict.keys() else 0 for token in temp_token_list]
#                 if sum(temp_token_freq) == 0:
#                     tokens.append(token2idx[seq[i]])
#                     i += 1
#                 else:
#                     selected_token = temp_token_list[np.argmax(temp_token_freq)]
#                     tokens.append(token2idx[selected_token])
#                     i += len(selected_token)
#         token_idx_list.append(tokens)
    
#     batch = []
#     for seq in token_idx_list:
#         input_ids = [token2idx['[CLS]']] + seq
#         output_labels = seq + [token2idx['[SEP]']]
#         mask_for_loss = [1] * len(input_ids)
#         n_pad1 = max_len - len(input_ids)
#         n_pad2 = max_len - len(output_labels)
#         if n_pad1>0:
#             input_ids.extend([0] * n_pad1)
#             mask_for_loss.extend([0] * n_pad1)
#         if n_pad2>0:
#             output_labels.extend([0] * n_pad2)

#         batch.append([input_ids, output_labels, mask_for_loss])
#     train_batch, validate_batch = shuffle_and_split_data(batch, ratio=0.01)
#     input_ids, output_labels, mask_for_loss = zip(*train_batch)
#     input_ids, output_labels, mask_for_loss = \
#     torch.LongTensor(input_ids), torch.LongTensor(output_labels),\
#     torch.LongTensor(mask_for_loss)
#     input_ids_v, output_labels_v, mask_for_loss_v = zip(*validate_batch)
#     input_ids_v, output_labels_v, mask_for_loss_v = \
#     torch.LongTensor(input_ids_v), torch.LongTensor(output_labels_v),\
#     torch.LongTensor(mask_for_loss_v)
    
#     train_dataset = Dataset_for_beta(input_ids, output_labels, mask_for_loss)
#     validate_dataset = Dataset_for_beta(input_ids_v, output_labels_v, mask_for_loss_v)


    # return train_dataset, validate_dataset, vocab_size

def make_data_for_gpt_pretrain(train_data_path, vocab_path, max_len):
    # 读取训练集txt文件
    train_data_df = pd.read_csv('/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/Data/peptide_pretrain/train_data_filtered.txt', sep='\t', header=None)
    train_seq_list = [seq for seq in train_data_df[0]]

    # 读取验证集txt文件
    valid_data_df = pd.read_csv('/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/Data/peptide_pretrain/valid.txt', sep='\t', header=None)
    valid_seq_list = [seq for seq in valid_data_df[0]]

    # token_df = pd.read_csv(vocab_path)
    # token_list = [token for token in token_df['token']]
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYWOBXJZ"]
    token_list = amino_acids  # 只使用氨基酸作为tokens
    token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[POS]': 3, '[NEG]': 4 }
    for i, w in enumerate(token_list):
        token2idx[w] = i + 5
    idx2token = {i: w for i, w in enumerate(token2idx)}
    vocab_size = len(token2idx)

    def process_seq_list(seq_list):
        token_idx_list = []
        for seq in seq_list:
            tokens = [token2idx[aa] for aa in seq[:50]]  # 只使用单个氨基酸作为token
            token_idx_list.append(tokens)

        batch = []
        for seq in token_idx_list:
            input_ids = [token2idx['[CLS]']] + seq
            # print(len(input_ids))
            output_labels = seq + [token2idx['[SEP]']]
            # print(len(output_labels))
            mask_for_loss = [1] * len(input_ids)
            n_pad1 = max_len - len(input_ids)
            n_pad2 = max_len - len(output_labels)
            if n_pad1 > 0:
                input_ids.extend([0] * n_pad1)
                mask_for_loss.extend([0] * n_pad1)
            if n_pad2 > 0:
                output_labels.extend([0] * n_pad2)

            batch.append([input_ids, output_labels, mask_for_loss])

        return batch

    train_batch = process_seq_list(train_seq_list)
    validate_batch = process_seq_list(valid_seq_list)

    input_ids, output_labels, mask_for_loss = zip(*train_batch)
    print("Number of input sequences:", len(input_ids))
    print("Length of the first input sequence:", len(input_ids[0]))
    input_ids, output_labels, mask_for_loss = \
        torch.LongTensor(input_ids), torch.LongTensor(output_labels), \
        torch.LongTensor(mask_for_loss)
    input_ids_v, output_labels_v, mask_for_loss_v = zip(*validate_batch)
    input_ids_v, output_labels_v, mask_for_loss_v = \
        torch.LongTensor(input_ids_v), torch.LongTensor(output_labels_v), \
        torch.LongTensor(mask_for_loss_v)

    train_dataset = Dataset_for_beta(input_ids, output_labels, mask_for_loss)
    validate_dataset = Dataset_for_beta(input_ids_v, output_labels_v, mask_for_loss_v)

    return train_dataset, validate_dataset, vocab_size

def make_data_for_gpt_finetune(train_data_path, vocab_path, max_len):
    # 读取正样本数据集
    pos_data_df = pd.read_csv('/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/Data/AMP/AMP.csv', sep=',', header=None)
    pos_seq_list = [seq for seq in pos_data_df[0]]

    # 读取负样本数据集
    neg_data_df = pd.read_csv('/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/Data/AMP/nonAMP.csv', sep=',', header=None)
    neg_seq_list = [seq for seq in neg_data_df[0]]

    # Token列表，仅使用氨基酸作为tokens
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYWOBXJZ"]
    token_list = amino_acids
    token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[POS]': 3, '[NEG]': 4}
    for i, w in enumerate(token_list):
        token2idx[w] = i + 5
    idx2token = {i: w for i, w in enumerate(token2idx.items())}
    vocab_size = len(token2idx)

    def process_seq_list(seq_list, label_token, label_value):
        token_idx_list = []
        labels = []
        for seq in seq_list:
            tokens = [token2idx[aa] for aa in seq[:50]]  # 留出位置给[CLS]和[POS]/[NEG]
            token_idx_list.append([label_token] + tokens)
            labels.append(label_value)

        batch = []
        for i, seq in enumerate(token_idx_list):
            input_ids = seq
            output_labels = seq[1:] + [token2idx['[SEP]']]  # 输出标签不包括[CLS]和[POS]/[NEG]
            mask_for_loss = [1] * len(output_labels)
            n_pad1 = max_len - len(input_ids)
            n_pad2 = max_len - len(output_labels)
            if n_pad1 > 0:
                input_ids.extend([0] * n_pad1)
            if n_pad2 > 0:
                output_labels.extend([0] * n_pad2)
                mask_for_loss.extend([0] * n_pad2)

            batch.append([input_ids, output_labels, mask_for_loss])

        input_ids, output_labels, mask_for_loss = zip(*batch)
        input_ids, output_labels, mask_for_loss = \
            torch.LongTensor(input_ids), torch.LongTensor(output_labels), \
            torch.LongTensor(mask_for_loss)

        return input_ids, output_labels, mask_for_loss, torch.LongTensor(labels)

    input_ids_pos, output_labels_pos, mask_for_loss_pos, labels_pos = process_seq_list(pos_seq_list, token2idx['[POS]'], 1)
    input_ids_neg, output_labels_neg, mask_for_loss_neg, labels_neg = process_seq_list(neg_seq_list, token2idx['[NEG]'], 0)

    input_ids = torch.cat((input_ids_pos, input_ids_neg), dim=0)
    output_labels = torch.cat((output_labels_pos, output_labels_neg), dim=0)
    mask_for_loss = torch.cat((mask_for_loss_pos, mask_for_loss_neg), dim=0)
    labels = torch.cat((labels_pos, labels_neg), dim=0)
    perm = torch.randperm(input_ids.size(0))
    input_ids = input_ids[perm]
    output_labels = output_labels[perm]
    mask_for_loss = mask_for_loss[perm]
    labels = labels[perm]
    # 将labels保存到文件
    # with open("labels.txt", "w") as f:
    #     for label in labels:
    #         f.write(str(label) + "\n")
    
    # print('labels',labels)

    # 划分训练集和验证集
    dataset_size = len(input_ids)
    indices = list(range(dataset_size))
    split = int(np.floor(0.05 * dataset_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset = Dataset_for_fintune(input_ids[train_indices], output_labels[train_indices], mask_for_loss[train_indices], labels[train_indices])
    validate_dataset = Dataset_for_fintune(input_ids[val_indices], output_labels[val_indices], mask_for_loss[val_indices], labels[val_indices])

    return train_dataset, validate_dataset, vocab_size

class Dataset_for_regression(Data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]

def make_data_for_regression(data_path, vocab_path, max_len):
    train_data_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/Data/AMP/train_reg.csv'
    # 读取训练集
    train_data_df = pd.read_csv(train_data_path, header=None)
    train_seq_list = [seq for seq in train_data_df[0]]
    train_labels = train_data_df[1].values
    valid_data_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/Data/AMP/test_reg.csv'
    # 读取验证集
    valid_data_df = pd.read_csv(valid_data_path, header=None)
    valid_seq_list = [seq for seq in valid_data_df[0]]
    valid_labels = valid_data_df[1].values

    # 读取词汇表
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYWOBXJZ"]
    token_list = amino_acids  # 只使用氨基酸作为tokens
    token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[POS]': 3, '[NEG]': 4 }
    for i, w in enumerate(token_list):
        token2idx[w] = i + 5
    idx2token = {i: w for i, w in enumerate(token2idx)}
    vocab_size = len(token2idx)

    def process_seq_list(seq_list):
        token_idx_list = []
        for seq in seq_list:
            tokens = [token2idx[aa] for aa in seq[:max_len-2]]  # 预留位置给 [CLS] 和 [SEP]
            token_idx_list.append(tokens)

        batch = []
        for seq in token_idx_list:
            input_ids = [token2idx['[CLS]']] + seq + [token2idx['[SEP]']]
            n_pad = max_len - len(input_ids)
            if n_pad > 0:
                input_ids.extend([token2idx['[PAD]']] * n_pad)
            batch.append(input_ids)

        return batch

    train_batch = process_seq_list(train_seq_list)
    validate_batch = process_seq_list(valid_seq_list)

    input_ids = torch.LongTensor(train_batch)
    labels = torch.FloatTensor(train_labels)  # 回归任务的标签是浮点数
    input_ids_v = torch.LongTensor(validate_batch)
    labels_v = torch.FloatTensor(valid_labels)  # 回归任务的标签是浮点数

    train_dataset = Dataset_for_regression(input_ids, labels)
    validate_dataset = Dataset_for_regression(input_ids_v, labels_v)

    return train_dataset, validate_dataset, vocab_size


def make_data_for_regression_without_labels(data_path, vocab_path, max_len):
    # 加载词汇表
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYWOBXJZ"]
    token_list = amino_acids  # 只使用氨基酸作为tokens
    token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[POS]': 3, '[NEG]': 4 }
    for i, w in enumerate(token_list):
        token2idx[w] = i + 5
    idx2token = {i: w for i, w in enumerate(token2idx)}
    vocab_size = len(token2idx)

    # 加载序列数据
    with open(data_path, "r") as f:
        sequences = [line.strip() for line in f.readlines()]

    # 将序列转换为模型可以接受的格式
    def process_seq_list(seq_list):
        token_idx_list = []
        for seq in seq_list:
            tokens = [token2idx[aa] for aa in seq[:max_len-2]]  # 预留位置给 [CLS] 和 [SEP]
            token_idx_list.append(tokens)

        batch = []
        for seq in token_idx_list:
            input_ids = [token2idx['[CLS]']] + seq + [token2idx['[SEP]']]
            n_pad = max_len - len(input_ids)
            if n_pad > 0:
                input_ids.extend([token2idx['[PAD]']] * n_pad)
            batch.append(input_ids)

        return batch

    test_batch = process_seq_list(sequences)

    input_ids = torch.LongTensor(test_batch)
    print(input_ids.type)

    test_dataset = Data.TensorDataset(input_ids)

    return test_dataset
        





    

    
        


        



    

