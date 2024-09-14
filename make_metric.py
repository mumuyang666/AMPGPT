import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from GPT import GPT_Model
from Data_prepare import make_data_for_gpt_pretrain
def calculate_average_cross_entropy(model, data_loader):
    model.eval()
    total_cross_entropy = 0
    total_samples = 0
    for batch in data_loader:
        input_ids, output_labels, mask_for_loss = batch
        device = next(model.parameters()).device
        input_ids, output_labels, mask_for_loss = input_ids.to(device), output_labels.to(device), mask_for_loss.to(device)
        with torch.no_grad():
            _, logits ,_ = model(input_ids)
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), output_labels.view(-1))
            loss = loss.view(output_labels.size())
            loss = (loss * mask_for_loss.float()).sum()
            total_cross_entropy += loss.item()
            total_samples += mask_for_loss.sum().item()
    average_cross_entropy = total_cross_entropy / total_samples
    return average_cross_entropy
def load_model(model_weights_path, vocab_size, d_model):
    model = GPT_Model_finetune(vocab_size, d_model)
    model.load_state_dict(torch.load(model_weights_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model
model_weights_path = '/aceph/louisyuzhao/buddy2/linyangxiao/AMP/AMPGPT/model/gpt_pep_pretrain_v0_Epoch_7.pth'
# 构造训练集和验证集

d_model = 768  # 可根据实际情况修改
train_data_path  = './'
vocab_path = './'
max_len = 52
train_dataset, validate_dataset, vocab_size = make_data_for_gpt_pretrain(train_data_path, vocab_path, max_len)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validate_data_loader = DataLoader(validate_dataset, batch_size=32, shuffle=True)
model = load_model(model_weights_path, vocab_size, d_model)
# 计算训练集和验证集上的平均交叉熵
train_average_cross_entropy = calculate_average_cross_entropy(model, train_data_loader)
validate_average_cross_entropy = calculate_average_cross_entropy(model, validate_data_loader)
print(f"训练集上的平均交叉熵: {train_average_cross_entropy}")
print(f"验证集上的平均交叉熵: {validate_average_cross_entropy}")