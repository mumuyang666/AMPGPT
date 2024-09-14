import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from BERT import set_seed
import argparse
# from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup
from Data_prepare import make_data_for_gpt_finetune


def get_attn_pad_mask(seq_q,seq_k):
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size,len_q,len_k)

def get_attn_subsequence_mask(seq):
    attn_shape=[seq.size(0),seq.size(1),seq.size(1)] #seq:b*tgt_len
    subsequence_mask=np.triu(np.ones(attn_shape),k=1)
    subsequence_mask=torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(seq.device)
    return subsequence_mask

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, maxlen=52, d_model=768):
        super().__init__()
        self.pos_embedding = nn.Embedding(maxlen, d_model)  
        self.token_embedding = nn.Embedding(vocab_size, d_model)  
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        pos = pos.to(x.device)
        embedding = self.token_embedding(x) + self.pos_embedding(pos)
        return self.drop(self.norm(embedding))


class Feed_Forward(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        x = self.linear2(self.relu(self.linear1(x)))
        x = self.drop(x)
        x = self.layer_norm(x+residual)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.drop = nn.Dropout(0.3)
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = self.drop(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.dense = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.dense(context)
        return self.norm(output + residual) # output: [batch_size, seq_len, d_model]


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block1 = MultiHeadAttention()
        self.feed_forward = Feed_Forward()

    def forward(self, input, mask):
        x = self.attention_block1(input, input, input, mask)
        output = self.feed_forward(x)
        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, layer_num=8):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size)
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(layer_num)])

    def forward(self, input):
        dec_self_attn_pad_mask = get_attn_pad_mask(input, input)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(input)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)
        emb = self.embedding(input)
        for layer in self.layers:
            emb = layer(emb, dec_self_attn_mask)
        return emb

import torch.nn.functional as F

def contrastive_loss(features, labels, margin=5.0):
    batch_size = features.size(0)
    dist_matrix = torch.cdist(features, features, p=2)  # 计算每对样本之间的欧氏距离
    labels = labels.unsqueeze(1)  # (B, 1)
    labels_t = labels.t()  # (1, B)
    mask = labels.eq(labels_t).float() 
        # 将labels保存到文件
    # with open("model_labels.txt", "w") as f:
    #     for label in labels:
    #         f.write(str(label) + "\n")
    
    # print('mask',mask) # 生成标签相同（1）和不同（0）的掩码

    pos_distances = dist_matrix * mask 
    # print('pos_distances',pos_distances) # 标签相同的距离
    # mask_n = 1-mask
    # print('mask_n',mask_n)
    neg_distances = dist_matrix * (1 - mask)  # 标签不同的距离
    # print('neg_distances',neg_distances)
    # 计算正样本对和负样本对的损失
    loss_pos = torch.clamp(pos_distances - margin, min=0.0)
    loss_neg = torch.clamp(margin - neg_distances, min=0.0)* (1 - mask)
    # print('pos',loss_pos)
    # print('neg',loss_neg)

    loss = loss_pos + loss_neg
    return loss.mean()
class GPT_Model_finetune(nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        self.decoder = Decoder(vocab_size)
        self.cls = nn.Linear(d_model, vocab_size)
        # self.discriminator = nn.Linear(d_model, 2)  # 用于分类抗菌肽和非抗菌肽

    def forward(self, input):
        decoder_out = self.decoder(input)
        pre = self.cls(decoder_out)
        discriminative_features = decoder_out.mean(dim=1)  # 平均池化
        # discriminative_output = self.discriminator(discriminative_features)
        return decoder_out, pre, discriminative_features

def create_parser():
    parser = argparse.ArgumentParser(description="GPT pretrain",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",dest="data_path",type=str,help="The data file in .csv format.",required=False)
    parser.add_argument("--vocab_path",dest="vocab_path",type=str,help="The vocab file in .csv format.",required=False)
    parser.add_argument("--pretrained_path", dest="pretrained_path", type=str, help="the path to load pretrained model.", required=False)
    parser.add_argument("--model_path",dest="model_path",type=str, help="the path to save model.", required=True)
    parser.add_argument("--epoch",dest="epoch",type=int,help="training epoch",default=10, required=False)
    parser.add_argument("--maxlen",dest="maxlen",type=int,help="maxlen of seq",default=52, required=False)
    parser.add_argument("--batch_size",dest="batch_size",type=int,help="batch_size",default=32, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=5e-5, required=False)
    parser.add_argument("--random_seed",type=int, dest="random_seed", default=0, help="seed for reproductbility",required=False)
    args = parser.parse_args()
    return args
from tqdm import tqdm

def train_step(model, data_loader, criterion, contrastive_criterion, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, output_labels, mask_for_loss, labels in data_loader:
        input_ids, output_labels, mask_for_loss, labels = input_ids.to(device), output_labels.to(device), mask_for_loss.to(device), labels.to(device)
        optimizer.zero_grad()
        _, logits_lm, discriminative_features = model(input_ids)
        loss_lm = criterion(logits_lm.transpose(1, 2), output_labels)
        loss_lm = loss_lm * mask_for_loss
        loss_lm = loss_lm.float().mean()
        loss_contrastive = contrastive_criterion(discriminative_features, labels)
        # loss = loss_lm + 0.5 * loss_contrastive
        loss = loss_lm
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def validate_step(model, data_loader, criterion, contrastive_criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, output_labels, mask_for_loss, labels in data_loader:
            input_ids, output_labels, mask_for_loss, labels = input_ids.to(device), output_labels.to(device), mask_for_loss.to(device), labels.to(device)
            _, logits_lm, discriminative_features = model(input_ids)
            loss_lm = criterion(logits_lm.transpose(1, 2), output_labels)
            loss_lm = loss_lm * mask_for_loss
            loss_lm = loss_lm.float().mean()
            loss_contrastive = contrastive_criterion(discriminative_features, labels)
            # loss = loss_lm + 0.5* loss_contrastive
            loss = loss_lm
            total_loss += loss.item()
    return total_loss / len(data_loader)
def main(args):
    set_seed(args.random_seed)
    train_dataset, validate_dataset, vocab_size = make_data_for_gpt_finetune(args.data_path, args.vocab_path, args.maxlen)
    data_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT_Model_finetune(vocab_size)
    model.to(device)

    # 加载预训练权重
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path))
        print(f"Loaded pretrained model from {args.pretrained_path}")

    criterion = nn.CrossEntropyLoss(reduction='none')
    contrastive_criterion = contrastive_loss
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(data_loader) * args.epoch
    warmup_steps = 0.1 * total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_losses = []
    best_train_loss = float('inf')
    best_train_epoch_model = None

    for epoch in range(args.epoch):
        train_loss = train_step(model, data_loader, criterion, contrastive_criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epoch}, Training Loss: {train_loss:.4f}")
        
        val_loss = validate_step(model, validate_dataloader, criterion, contrastive_criterion, device)
        print(f"Epoch {epoch + 1}/{args.epoch}, Validation Loss: {val_loss:.4f}")

        # 保存验证集损失最小的模型
        if len(best_val_losses) < 3:
            best_val_losses.append((val_loss, epoch, model.state_dict()))
            best_val_losses = sorted(best_val_losses, key=lambda x: x[0])
        elif val_loss < best_val_losses[-1][0]:
            best_val_losses[-1] = (val_loss, epoch, model.state_dict())
            best_val_losses = sorted(best_val_losses, key=lambda x: x[0])

        # 保存训练集损失最小的模型
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch_model = (epoch, model.state_dict())

    # 保存验证集损失最小的3个epoch
    for i, (loss, epoch, state_dict) in enumerate(best_val_losses):
        torch.save(state_dict, f"{args.model_path}_best_val_epoch_{epoch + 1}.pt")
        print(f"Saved best validation model {i + 1} with loss {loss:.4f} at epoch {epoch + 1}")

    # 保存训练集损失最小的模型
    if best_train_epoch_model is not None:
        epoch, state_dict = best_train_epoch_model
        torch.save(state_dict, f"{args.model_path}_best_train_epoch_{epoch + 1}.pt")
        print(f"Saved best training model with loss {best_train_loss:.4f} at epoch {epoch + 1}")

    # 保存训练集最后一个epoch的模型
    torch.save(model.state_dict(), f"{args.model_path}_last_epoch.pt")
    print(f"Saved model from the last epoch at {args.epoch}")

if __name__ == "__main__":
    args = create_parser()
    main(args)
# def main(args):
#     set_seed(args.random_seed)
#     dataset, validate_dataset, vocab_size = make_data_for_gpt_finetune(args.data_path, args.vocab_path, args.maxlen)
#     data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#     validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = GPT_Model(vocab_size)
#     model.to(device)
#     criterion = nn.CrossEntropyLoss(reduction='none')
#     optimizer = AdamW(model.parameters(), lr=args.learning_rate)
#     total_steps = len(data_loader) * args.epoch
#     warmup_steps = 0.1 * total_steps
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

#     best_val_losses = [(float("inf"), None) for _ in range(3)]

#     for epoch in range(args.epoch):
#         model.train()
#         loss = 0
#         for input_ids, output_labels, mask_for_loss in tqdm(data_loader, desc=f"Training epoch {epoch}"):
#             input_ids, output_labels, mask_for_loss = input_ids.to(device), output_labels.to(device), mask_for_loss.to(device)
#             optimizer.zero_grad()
#             _, logits_lm = model(input_ids)
#             loss_lm = criterion(logits_lm.transpose(1, 2), output_labels) 
#             loss_lm = loss_lm * mask_for_loss
#             loss_lm = (loss_lm.float()).mean()
#             loss = loss + loss_lm.item()
#             loss_lm.backward()
#             optimizer.step()
#             scheduler.step()
#         print(f"epoch:{epoch},done! loss:{loss}")

#         model.eval()
#         loss_v = 0
#         with torch.no_grad():
#             for input_ids, output_labels, mask_for_loss in validate_dataloader:
#                 input_ids, output_labels, mask_for_loss = input_ids.to(device), output_labels.to(device), mask_for_loss.to(device)
#                 _, logits_lm = model(input_ids)
#                 loss_lm = criterion(logits_lm.transpose(1, 2), output_labels) 
#                 loss_lm = loss_lm * mask_for_loss
#                 loss_lm = (loss_lm.float()).mean()
#                 loss_v = loss_v + loss_lm.item()
#             print(f"loss in validate_dataset:{loss_v}")

#         if loss_v < max(best_val_losses, key=lambda x: x[0])[0]:
#             # 删除之前的权重文件
#             _, old_model_path = max(best_val_losses, key=lambda x: x[0])
#             if old_model_path is not None:
#                 os.remove(old_model_path)

#             # 保存当前模型权重
#             current_model_path = f"{args.model_path}_epoch_{epoch}.pth"
#             torch.save(model.state_dict(), current_model_path)

#             # 更新最低验证损失值
#             best_val_losses.remove(max(best_val_losses, key=lambda x: x[0]))
#             best_val_losses.append((loss_v, current_model_path))

# if __name__=="__main__":
#     args=create_parser()
#     main(args)