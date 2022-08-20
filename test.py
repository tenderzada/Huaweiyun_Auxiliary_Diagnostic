# coding: UTF-8
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

from data_loader import DADataSet
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser(description='The model for disease diagnosis.')
    parser.add_argument('--data_path', default='./train.json', type=str)
    parser.add_argument('--batch_size', default=32, type=int, help='the batch size of dataset')
    parser.add_argument('--bert_path',
                        default='./chinese-bert-wwm-ext')
    parser.add_argument('--threshold', default=0.5, type=float, help='the threshold of disease predictive')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    return parser.parse_args()


def inference():
    args = get_args()
    test_dataset = DADataSet(args)
    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=args.batch_size,
                           shuffle=True
                           )
    model = BaselineModel(args.bert_path)
    model.load_state_dict(torch.load('baseline_model.pt', map_location='cpu'))
    model.eval()
    model.to(args.device)
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, (text, mask, _) in enumerate(test_iter):
            outputs = model(text, mask)
            pred_labels = outputs.cpu().numpy()
            pred_labels = [1 if pred > args.threshold else 0 for pred in pred_labels[0]]
            if i == 0:
                predict_all = pred_labels
            else:
                predict_all = np.vstack((predict_all, pred_labels))
    print('return result...')
    print('The first emr result :{}'.format(predict_all[0]))


if __name__ == '__main__':
    inference()
