
""" Script for training and testing the model. """

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertModel, BertConfig, DistilBertTokenizer
from types import SimpleNamespace
from torch.utils.data import DataLoader
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import mlflow
from codecarbon import EmissionsTracker

from features import build_features


MODULE_PATH = os.path.dirname(__file__)

MODEL_NAME = "distilbert-base-cased"


tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, max_length=512, padding=True,
                                                truncation = True, return_tensors="pt")

args = SimpleNamespace(
    batch_size = 1,             # training and valid batch size
    test_batch_size = 1,        # batch size for testing
    epochs = 15,                # maximum number of epochs to train
    lr = 0.001,                 # learning rate
    momentum = 0.9,             # SGD momentum, for SGD only
    optimizer = 'adam',         # optimization method: sgd | adam
    log_interval = 5,           # number of batches before logging training status
    patience = 5,               # number of epochs of no loss improvement before stop training
    checkpoint = '.',           #
    seed = 42,                  # checkpoints directory
    train = True,               # train before testing
    cuda = True,                # use gpu
    num_workers = 2,            # number of subprocesses to use for data loading
    adapter_hidden_size = 32,   #
    acc_steps = 20              #
)


class Loader(torch.utils.data.Dataset):
    def __init__(self, comments, sentiments):
        self.data = tokenizer(comments, padding=True, truncation = True,
                              max_length=512,return_tensors="pt")['input_ids']
        self.target = sentiments

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        return data, target

    def __len__(self):
        return len(self.target)


class DistilBERTforSentiment(nn.Module):
    def __init__(self, adapter_hidden_size=args.adapter_hidden_size):
        super().__init__()

        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)

        hidden_size = self.distilbert.config.hidden_size

        self.adaptor = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(adapter_hidden_size, hidden_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(adapter_hidden_size, 2),
        )

    def forward(self, inputs):
        outputs = self.distilbert(input_ids = inputs, return_dict=False)
        # B x seq_length x H
        x = self.adaptor(outputs[0])

        x,_ = x.max(dim=1)
        # B x H

        results = self.classifier(x)
        return results


def train_one_epoch(trainloader, model, criterion, optimizer, epoch_index, cuda,max_norm=1):
    model.train()
    running_loss = 0
    accumulation_steps = args.acc_steps # effective  batch
    for i, (input_ids,target) in enumerate(trainloader, 0):
        if cuda:
            input_ids, target = input_ids.cuda(), target.cuda()
        output = model(input_ids)

        loss = criterion(output, target)
        #print("output ", output," target: ", target, " ", i)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if (i+1) % accumulation_steps == 0:
            optimizer.step()                 # Now we can do an optimizer step
            optimizer.zero_grad()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(trainloader) + i + 1
            print('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return running_loss / i


def test_one_epoch(test_loader,model,criterion, cuda, avg_loss):
    running_vloss = 0.0
    acc = 0
    _p = 0.0000000001 # prediction: not pos, target: pos
    _n = 0.0000000001 # prediction: not neg, target: neg
    _t = 0.0000000001 # prediction: not neutral, target: neutral
    pp = 0.0000000001 # prediction: pos, target: pos
    nn = 0.0000000001 # prediction: neg, target: neg
    tt = 0.0000000001 # prediction: neutral, target: neutral
    p_ = 0.0000000001 # prediction: pos, target: not pos
    n_ = 0.0000000001 # prediction: neg, target: not neg
    t_ = 0.0000000001 # prediction: neutral, target: not neutral
    for i, (input_ids,target) in enumerate(test_loader, 0):
        if cuda:
            input_ids, target = input_ids.cuda(), target.cuda()
        output = model(input_ids)
        loss = criterion(output, target)
        running_vloss += loss
        obj = torch.Tensor.int(target.argmax())
        out = torch.Tensor.int(output.argmax())
        if out == obj:
            acc += 1
            if obj == 0:
                nn += 1
            else:
                pp += 1
            #else:
            #    pp += 1
        else:
            if obj == 0:
                _n += 1
            else:
                _p += 1
            #else:
            #   _p += 1
            if out == 0:
                n_ += 1
            else:
                p_ += 1
            #else:
            #    p_ += 1

    prec_pos = pp/(pp + p_)
    prec_neg = nn/(nn + n_)
    #prec_neu = tt/(tt + t_)
    rec_pos = pp/(pp + _p)
    rec_neg = nn/(nn + _n)
    #rec_neu = tt/(tt + _t)
    avg_vloss = running_vloss / (i + 1)
    acc = acc/(i+1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('Accuracy = {}'.format(acc))
    return acc, avg_vloss, rec_neg, 0, rec_pos, prec_neg, 0, prec_pos


def train_model(X_train, X_valid, y_train, y_valid):
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))

    # build model
    model = DistilBERTforSentiment(adapter_hidden_size=args.adapter_hidden_size)

    for param in model.distilbert.parameters():
        param.requires_grad = False

    if args.cuda:
        model.cuda()

    # Define criterion
    criterion = nn.CrossEntropyLoss()

    print("Preparing training set...")
    training_set = Loader(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                            num_workers=0, shuffle = True)
    print("Preparing validation set...")

    valid_set = Loader(X_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.test_batch_size,
                                            num_workers=0, shuffle = True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epoch = 0
    best_valid_loss = 9999
    experiment_name = "change hidd_siz64"
    mlflow.set_tracking_uri("https://dagshub.com/danielgonzalbez/taed.mlflow")
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'danielgonzalbez'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'iqh601lm'
    print("Creating experiment...")
    experiment_id = mlflow.create_experiment(experiment_name)
    print("Starting the experiment...")

    tracker = EmissionsTracker()
    tracker.start()
    with mlflow.start_run(experiment_id = experiment_id):
        mlflow.log_param("batch_size", args.acc_steps)
        mlflow.log_param("adapter_hidden_size", args.adapter_hidden_size)
        mlflow.log_param("momentum", args.momentum)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("epochs", args.epochs)
        while epoch < args.epochs + 1:
            train_loss = train_one_epoch(train_loader, model, criterion,
                                         optimizer, epoch, args.cuda)
            acc, valid_loss, rec_neg, rec_neu, rec_pos, prec_neg, prec_neu, prec_pos = \
                test_one_epoch(valid_loader, model, criterion, args.cuda, train_loss)
            if not os.path.isdir(args.checkpoint):
                os.mkdir(args.checkpoint)
            torch.save(model.state_dict(), './{}/model{:03d}.pt'.format(args.checkpoint, epoch))
            if valid_loss <= best_valid_loss:
                print('Saving state')
                best_valid_loss = valid_loss
                best_epoch = epoch
                mlflow.log_artifact('./{}/model{:03d}.pt'.format(args.checkpoint, epoch))
                state = {
                    'valid_loss': valid_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir(args.checkpoint):
                    os.mkdir(args.checkpoint)
                torch.save(state, './{}/ckpt.pt'.format(args.checkpoint))

                print ('logging accuracy...')
                mlflow.log_metric("accuracy", acc)
                print('logging loss...')
                mlflow.log_metric("loss", valid_loss)
                mlflow.log_metric("Negative recall", rec_neg)
                #mlflow.log_metric("Neutral recall", rec_neu)
                mlflow.log_metric("Positive recall", rec_pos)
                mlflow.log_metric("Negative precision", prec_neg)
                #mlflow.log_metric("Neutral precision", prec_neu)
                mlflow.log_metric("Positive precision", prec_pos)
                mlflow.log_metric("Best epoch", epoch)
            print("End epoch ", epoch)

            epoch += 1
        emissions = tracker.stop()
        mlflow.log_metric("Emissions", emissions)


def test_model(X_test, y_test):
    test_set = Loader(X_test, y_test) 
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                              num_workers=0, shuffle = True)
    acc = 0
    num1 = 0
    num0 = 0

    # i_j = Predicted i, Target j
    _p = 0.0000000001 # prediction: not pos, target: pos
    _n = 0.0000000001 # prediction: not neg, target: neg
    _t = 0.0000000001 # prediction: not neutral, target: neutral
    pp = 0.0000000001 # prediction: pos, target: pos
    nn = 0.0000000001 # prediction: neg, target: neg
    tt = 0.0000000001 # prediction: neutral, target: neutral
    p_ = 0.0000000001 # prediction: pos, target: not pos
    n_ = 0.0000000001 # prediction: neg, target: not neg
    t_ = 0.0000000001 # prediction: neutral, target: not neutral

    for i, (input_ids, target) in enumerate(test_loader):
        output = model(input_ids)
        obj = torch.Tensor.int(target.argmax())
        out = torch.Tensor.int(output.argmax())
        if out == obj:
            acc += 1
            if obj == 0:
                nn += 1
            else:
                pp += 1
            #else:
            #    pp += 1
        else:
            if obj == 0:
                _n += 1
            else:
                _p += 1
            #else:
            #   _p += 1
            if out == 0:
                n_ += 1
            else:
                p_ += 1
            #else:
            #    p_ += 1
        if(i%100 == 0):
            print(i, " ",acc/(i+1))
            
    prec_pos = pp/(pp + p_)
    prec_neg = nn/(nn + n_)
    #prec_neu = tt/(tt + t_)
    rec_pos = pp/(pp + _p)
    rec_neg = nn/(nn + _n)
    #rec_neu = tt/(tt + _t)
    print("FINAL ACCURACY: ", acc/i)
    print("Positive Recall: ", rec_pos)
    print("Negative Recall: ", rec_neg)
    #print("Neutral Recall: ", rec_neu)
    print("Negative Precision: ", prec_neg)
    #print("Neutral Precision: ", prec_neu)
    print("Positive Precision: ", prec_pos)


    print("F1-Score Positive: ", (2*prec_pos*rec_pos/(prec_pos+rec_pos)))
    print("F1-Score Negative: ", (2*prec_neg*rec_neg/(prec_neg+rec_neg)))



def main():
    data_clean = "../data/processed/comments_clean.csv"
    comments, target = build_features.preprocess(data_clean=data_clean,
                                                 text_col='Comment', target_col='Sentiment')
    X_train, X_test, y_train, y_test = train_test_split(comments, target,
                                                        test_size=0.2, random_state=args.seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.2, random_state=args.seed)
    train_model(X_train, X_valid, y_train, y_valid)
    test_model(X_test, y_test)


if __name__ == '__main__':
    main()
