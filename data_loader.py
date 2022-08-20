import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DADataSet(Dataset):

    def __init__(self, args):
        with open(args.data_path, 'r', encoding='utf8') as f:
            self.lines = f.readlines()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        with open("./labels.txt", 'r', encoding='utf8') as f:
            labels = f.readlines()
        label_icd = []
        label2id = {}
        for idx, label in enumerate(labels):
            label_icd.append(label.strip())
            label2id[label.strip()] = idx
        self.label2id = label2id

    def __getitem__(self, idx):
        item = self.lines[idx]
        data = json.loads(item)
        # present_illness = data["history_of_present_illness"]
        # present_illness = data["chief_complaint"] + data["history_of_present_illness"] + data ["past_history"] + data["physical_examination"]
        present_illness = data["history_of_present_illness"]
        input_ids, attn_mask = self._get_token(present_illness, 128)
        targets = data['diagnosis']
        label = self._get_label_ids(targets)
        return input_ids, attn_mask, label

    def __len__(self):
        return len(self.lines)

    def _get_token(self, content, pad_size):
        all_tokens = self.tokenizer.encode_plus(content, max_length=pad_size, padding="max_length", truncation=True)
        input_ids = torch.LongTensor(all_tokens['input_ids']).to(self.args.device)
        attention_mask = torch.LongTensor(all_tokens['attention_mask']).to(self.args.device)
        return input_ids, attention_mask

    def _get_label_ids(self, targets):
        label = [0] * len(self.label2id)
        for target in targets:
            if target in self.label2id:
                idx = self.label2id[target]
                label[idx] = 1
        label = torch.FloatTensor(label).to(self.args.device)
        return label
