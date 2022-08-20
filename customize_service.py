# -*- coding: utf-8 -*-
import logging
import os

import torch
import json
from transformers import BertTokenizer

from model_service.pytorch_model_service import PTServingBaseService
from model import BaselineModel

logger = logging.getLogger(__name__)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DaService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTServingBaseService, self).__init__(model_name, model_path)
        dir_path = os.path.dirname(os.path.realpath(model_path))
        bert_path = os.path.join(dir_path, 'chinese-bert-wwm-ext')
        self.model = BaselineModel(bert_path=bert_path)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        label_path = os.path.join(dir_path, 'labels.txt')
        self.id2label = []
        with open(label_path, 'r', encoding='utf8') as f:
            for line in f:
                self.id2label.append(line.strip())

    def _get_token(self, content, pad_size):
        all_tokens = self.tokenizer.encode_plus(content, max_length=pad_size, padding="max_length", truncation=True)
        input_ids = torch.LongTensor(all_tokens['input_ids']).to(DEVICE)
        attention_mask = torch.LongTensor(all_tokens['attention_mask']).to(DEVICE)
        return input_ids, attention_mask

    def _get_diagnosis(self, pred):
        pred_index = [i for i in range(len(pred)) if pred[i] == 1]
        pred_diagnosis = [self.id2label[index] for index in pred_index]
        return pred_diagnosis

    def _preprocess(self, data):
        # https两种请求形式
        # 1. form-data文件格式的请求对应：data = {"请求key值":{"文件名":<文件io>}}
        # 2. json格式对应：data = json.loads("接口传入的json体")
        data_dict = data.get('json_line')
        for v in data_dict.values():
            infer_dict = json.loads(v.read())
            return infer_dict

    def _inference(self, data):
        self.model.eval()
        emr_id = data.get('emr_id')
        text_data = data.get('history_of_present_illness')
        text, mask = self._get_token(text_data, 128)
        output = self.model(text.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE))
        result = {emr_id: output}
        return result

    def _postprocess(self, data):
        infer_output = None
        for k, v in data.items():
            pred_labels = v.cpu().detach().numpy()
            pred_labels = [1 if pred > 0.5 else 0 for pred in pred_labels[0]]
            pred_diagnosis = self._get_diagnosis(pred_labels)
            infer_output = {k: pred_diagnosis}
        return infer_output
