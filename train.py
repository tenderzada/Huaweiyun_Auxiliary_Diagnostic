# coding: UTF-8
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from torch.utils.data import DataLoader

from data_loader import DADataSet
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser(description='The model for disease diagnosis.')
    parser.add_argument('--data_path', default='./train.json', type=str)
    parser.add_argument('--num_epochs', default=50, type=int, help='the epoch of train')
    parser.add_argument('--batch_size', default=32, type=int, help='the batch size of dataset')
    parser.add_argument('--lr', default=5e-5, type=float, help='the learning rate of bert')
    parser.add_argument('--bert_path',
                        default='./chinese-bert-wwm-ext')
    parser.add_argument('--warm_up', default=0.01)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    return parser.parse_args()


def main():
    args = get_args()
    train_dataset = DADataSet(args)
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True
                            )

    model = BaselineModel(args.bert_path).to(args.device)
    model.train()
    num_training_steps = len(train_iter) * args.num_epochs
    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=0.01)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=args.warm_up * num_training_steps,
                                                             num_training_steps=num_training_steps)
    for j in range(args.num_epochs):
        total_iter = 0
        total_loss = 0.
        for i, (text, mask, labels) in enumerate(tqdm(train_iter)):
            model.zero_grad()
            outputs = model(text, mask)
            loss_function = nn.BCELoss()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_iter += 1
            total_loss += loss.data
        avg_loss = total_loss / total_iter
        print(f'[Epoch]: {j}')
        print(f'[Avg_Loss]: {avg_loss}')
    torch.save(model.state_dict(), 'baseline_model.pt')
    print('Save Best Model...')


if __name__ == '__main__':
    main()
