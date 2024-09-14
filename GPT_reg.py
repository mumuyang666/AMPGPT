import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from BERT import set_seed
import argparse
import csv
# from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup
from Data_prepare import make_data_for_regression,make_data_for_regression_without_labels
from GPT import GPT_Model

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
from scipy.stats import pearsonr

def test_model(model, test_dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, output_labels in test_dataloader:
            input_ids, output_labels = input_ids.to(device), output_labels.to(device)
            predictions = model(input_ids)
            all_predictions.extend(predictions.squeeze().tolist())
            all_labels.extend(output_labels.tolist())

    r_value, _ = pearsonr(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)

    return r_value, r2, mse, mae
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
def predict(model, dataloader, device, sequences):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)  # 解包张量
            predictions = model(input_ids)
            all_predictions.extend(predictions.squeeze().tolist())

    return sequences, all_predictions

class GPT_Model_reg(GPT_Model):
    def __init__(self, vocab_size, d_model=768):
        super().__init__(vocab_size, d_model)  # 修改这里，回归层输出一个值
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, input):
        decoder_out = self.decoder(input)
        pre = self.regressor(decoder_out[:, -1, :])  # 只取序列最后一个时间步的输出进行回归
        return pre

def create_parser():
    parser = argparse.ArgumentParser(description="GPT pretrain",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",dest="data_path",type=str,help="The data file in .csv format.",required=False)
    parser.add_argument("--vocab_path",dest="vocab_path",type=str,help="The vocab file in .csv format.",required=False)
    parser.add_argument("--pretrained_path", dest="pretrained_path", type=str, help="the path to load pretrained model.", required=False)
    parser.add_argument("--model_path",dest="model_path",type=str, help="the path to save model.", required=False)
    parser.add_argument("--epoch",dest="epoch",type=int,help="training epoch",default=3, required=False)
    parser.add_argument("--maxlen",dest="maxlen",type=int,help="maxlen of seq",default=52, required=False)
    parser.add_argument("--batch_size",dest="batch_size",type=int,help="batch_size",default=32, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=5e-5, required=False)
    parser.add_argument("--random_seed",type=int, dest="random_seed", default=0, help="seed for reproductbility",required=False)
    args = parser.parse_args()
    return args
from tqdm import tqdm
def main(args):
    set_seed(args.random_seed)
    print(args.maxlen)
    dataset, validate_dataset, vocab_size = make_data_for_regression(args.data_path, args.vocab_path, args.maxlen)
    data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GPT_Model_reg(vocab_size)
    # model.to(device)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print('true')
        gpt_model_reg = GPT_Model_reg(vocab_size)

        # 加载预训练的GPT_Model参数
        gpt_model_reg.load_state_dict(torch.load(args.pretrained_path), strict=False)
    else :
        print('pretrain_fasle')
        gpt_model_reg = GPT_Model_reg(vocab_size)
    model = gpt_model_reg
    model.to(device)
    

    criterion = nn.MSELoss()  # 修改这里，使用均方误差损失
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(data_loader) * args.epoch
    warmup_steps = 0.1 * total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_losses = [(float("inf"), None) for _ in range(3)]
    loss_log_file = open("loss_reg_log.log", "w")

    for epoch in range(args.epoch):
        model.train()
        loss = 0
        for input_ids, output_labels in tqdm(data_loader, desc=f"Training epoch {epoch}"):
            input_ids, output_labels = input_ids.to(device), output_labels.to(device)
            optimizer.zero_grad()
            predictions = model(input_ids)
            loss_lm = criterion(predictions.squeeze(), output_labels.float())
            loss_log_file.write(f"Batchloss: {loss_lm.item()}\n")
            loss_log_file.flush()
            loss += loss_lm.item()
            loss_lm.backward()
            optimizer.step()
            scheduler.step()
        print(f"epoch: {epoch}, done! loss: {loss / len(data_loader)}")

        model.eval()
        loss_v = 0
        with torch.no_grad():
            for input_ids, output_labels in validate_dataloader:
                input_ids, output_labels = input_ids.to(device), output_labels.to(device)
                predictions = model(input_ids)
                loss_lm = criterion(predictions.squeeze(), output_labels.float())
                loss_v += loss_lm.item()
            print(f"loss in validate_dataset: {loss_v / len(validate_dataloader)}")

        if loss_v < max(best_val_losses, key=lambda x: x[0])[0]:
            _, old_model_path = max(best_val_losses, key=lambda x: x[0])
            if old_model_path is not None:
                os.remove(old_model_path)

            current_model_path = f"{args.model_path}_v0_reg_Epoch_{epoch}.pth"
            torch.save(model.state_dict(), current_model_path)

            best_val_losses.remove(max(best_val_losses, key=lambda x: x[0]))
            best_val_losses.append((loss_v, current_model_path))
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
def cal_metric(args):
# 在main函数中添加以下代码来进行测试
    dataset, validate_dataset, vocab_size = make_data_for_regression(args.data_path, args.vocab_path, args.maxlen)
    data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)
    # test_dataloader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = GPT_Model_reg(vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载预训练的权重
    pretrained_weights = torch.load('/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/model/gpt_pep_reg_v0_reg_Epoch_36.pth')

    # 将权重应用到模型
    model.load_state_dict(pretrained_weights)

    # r_value, r2, mse, mae = test_model(model, validate_dataloader, device)
    # print(f"Pearson R: {r_value}, R-Squared: {r2}, MSE: {mse}, MAE: {mae}")
    test_data_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/result/finetune_generated_sequences.txt'
    test_dataset = make_data_for_regression_without_labels(test_data_path, args.vocab_path, args.maxlen)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    with open(test_data_path, "r") as f:
        sequences = [line.strip() for line in f.readlines()]

    sequences, predictions = predict(model, test_dataloader, device, sequences)

    with open('/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/result/reg_infer.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Sequence', 'Prediction'])
        for seq, pred in zip(sequences, predictions):
            writer.writerow([seq, pred])
if __name__ == "__main__":
    args = create_parser()
    # main(args)
    cal_metric(args)
