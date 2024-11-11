from torch import nn
import torch
class concat_model_linear(nn.Module):
    def __init__(self,hidden_size):
        super(concat_model_linear,self).__init__()
        self.hidden_size = hidden_size
        self.linear= nn.Linear(2*hidden_size,hidden_size)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size,hidden_size)
    def forward(self,rel_img,rel_text):
        concat_feature= torch.cat((rel_img,rel_text),dim=1)
        concat_feature= self.linear(concat_feature)
        concat_feature= self.relu(concat_feature)
        concat_feature= self.linear2(concat_feature)
        return concat_feature