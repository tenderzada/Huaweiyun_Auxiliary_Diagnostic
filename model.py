import torch
import torch.nn as nn
from transformers import BertModel

class BaselineModel(nn.Module):
    def __init__(self, bert_path):
        super(BaselineModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(768, 52)
        # small model
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 52)
        # big model
        # self.fc1 = nn.Linear(1024, 128)
        # self.fc2 = nn.Linear(128, 52)

    def forward(self, text, mask):
        bert_out = self.bert(text, attention_mask=mask)[1]
        bert_out = self.dropout(bert_out)
        # out_linear = self.fc1(bert_out)
        out_linear_1 = self.fc1(bert_out)
        out_linear_2 = self.fc2(out_linear_1)
        output = torch.sigmoid(out_linear_2)
        return output

if __name__ == "__main__":
    model = BaselineModel()
    print(model)