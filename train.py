# coding: UTF-8
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from torch.utils.data import DataLoader

from data_loader import DADataSet
from model import BaselineModel

# divide dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


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

# unuse
def evaluate(y_true, y_pre): 
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true.cpu(), y_pre.cpu())
    # average='macro', calculate the average precision, recall, and F1 score of all categories.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    return acc, precision, recall, f1


def main():
    args = get_args()

    #train_dataset = DADataSet(args)
    dataset = DADataSet(args)
    trainset, validationset = train_test_split(dataset, test_size=0.2)

    # train_iter = DataLoader(dataset=train_dataset,
    #                         batch_size=args.batch_size,
    #                         shuffle=True
    #                         )
    train_iter = DataLoader(dataset=trainset,
                            batch_size=args.batch_size,
                            shuffle=True
                            )
    # the number of training data
    print(f"trainset length: {len(train_iter)*args.batch_size}")

    validation_iter = DataLoader(dataset=validationset,
                            # batch_size=args.batch_size,
                            batch_size=64,
                            shuffle=True
                            )
    # the number of validation data
    print(f"validationset length: {len(validation_iter)*64}")
                        
    model = BaselineModel(args.bert_path).to(args.device)
    # model.train()
    num_training_steps = len(train_iter) * args.num_epochs
    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=0.01)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=args.warm_up * num_training_steps,
                                                             num_training_steps=num_training_steps)
    for j in range(args.num_epochs):
        model.train()
        print(f"[Epoch]: {j}")
        print("training:")
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
        train_avg_loss = total_loss / total_iter
        print(f'[Train_Avg_Loss]: {train_avg_loss}')

        if ((j+1)%5==0):
            model.eval()
            total_iter = 0
            total_loss = 0
            # validation in the end of each 5 epoch
            with torch.no_grad():
                for i, (text, mask, labels) in enumerate(tqdm(validation_iter)):
                    outputs = model(text, mask)
                    loss_function = nn.BCELoss()
                    loss = loss_function(outputs, labels)
                    total_iter += 1
                    total_loss += loss
                validation_avg_loss = total_loss / total_iter
                print(f'[Validation_Avg_loss]: {validation_avg_loss}')

    torch.save(model.state_dict(), 'baseline_model.pt')
    print('Save Best Model...')


if __name__ == '__main__':
    main()
