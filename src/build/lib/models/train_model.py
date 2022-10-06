import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertModel, DistilBertTokenizer
from types import SimpleNamespace
import os
from sklearn.model_selection import train_test_split
import mlflow

from src import build_features


MODEL = "distilbert-base-multilingual-cased"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL, max_length=512, padding=True, truncation = True, return_tensors="pt")

args = SimpleNamespace(
    batch_size = 1,                             # training and valid batch size
    test_batch_size = 1,                        # batch size for testing
    epochs = 5,                                 # maximum number of epochs to train
    lr = 0.0005,                                # learning rate
    log_interval = 5,                           # how many batches to wait before logging training status
    patience = 5,                               # how many epochs of no loss improvement should we wait before stop training
    checkpoint = '.',                           # checkpoints directory
    seed = 42,                                  # random seed
    train = True,                               # train before testing
    cuda = True,                                # use gpu
    num_workers = 1,                            # how many subprocesses to use for data loading
    adapter_hidden_size = 16
)


class Loader(torch.utils.data.Dataset):
  def __init__(self, comments, sentiments):
    self.data= tokenizer(comments, padding=True, truncation = True, max_length=512,return_tensors="pt")['input_ids']
    self.target = sentiments

  def __getitem__(self, index):
    data = self.data[index]
    target = self.target[index]
    return data, target

  def __len__(self):
        return len(self.target)


class DistilBERTforSentiment(nn.Module):
    def __init__(self, adapter_hidden_size=16):
        super().__init__()

        self.distilbert = DistilBertModel.from_pretrained(MODEL)

        hidden_size = self.distilbert.config.hidden_size

        self.adaptor = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_size, hidden_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_size, 3),
        )

    def forward(self, inputs):
        outputs = self.distilbert(input_ids = inputs, return_dict=False)
        # B x seq_length x H
        x = self.adaptor(outputs[0])

        x,_ = x.max(dim=1)
        # B x H

        results = self.classifier(x)
        return results


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


def train_one_epoch(trainloader, model, criterion, optimizer, epoch_index, cuda,max_norm=1):
    model.train()
    running_loss = 0
    accumulation_steps = 40 # effective 40 batch
    for i, (input_ids,target) in enumerate(trainloader, 0):
        if cuda:
            input_ids, target = input_ids.cuda(), target.cuda()
        output = model(input_ids)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
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
    # best_vloss = 99999
    acc = 0
    for i, (input_ids,target) in enumerate(test_loader, 0):
        if cuda:
            input_ids, target = input_ids.cuda(), target.cuda()
        output = model(input_ids)
        loss = criterion(output, target)
        running_vloss += loss
        if output.argmax() == target.argmax():
            acc += 1

    avg_vloss = running_vloss / (i + 1)
    acc = acc/(i+1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('Accuracy= {}'.format(acc))
    return acc, avg_vloss

data_file = "../data/raw/comments.csv"
comments, target = build_features.preprocess(data_file, text_col='Comments', target_col='Sentiment')
X_train, X_test, y_train, y_test = train_test_split(comments, target, test_size=0.33, random_state=args.seed)

print("Preparing training set...")
training_set = Loader(X_train, y_train)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)

print("Preparing validation set...")
test_set = Loader(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                          shuffle=True, num_workers=args.num_workers)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
epoch = 0
best_valid_loss = 9999
experiment_name = "Hid_Adapter=90"
#mlflow.set_tracking_uri('http://127.0.0.1:5000/')
print("Creating experiment...")
experiment_id = mlflow.create_experiment(experiment_name)
print("Starting the experiment...")
with mlflow.start_run(experiment_id = experiment_id):
  while (epoch < args.epochs + 1):
      train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args.cuda)
      acc, valid_loss = test_one_epoch(test_loader, model, criterion, args.cuda, train_loss)
      if not os.path.isdir(args.checkpoint):
          os.mkdir(args.checkpoint)
      torch.save(model.state_dict(), './{}/model{:03d}.pt'.format(args.checkpoint, epoch))
      if valid_loss <= best_valid_loss:
          print('Saving state')
          best_valid_loss = valid_loss
          best_epoch = epoch
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
      print("End epoch ", epoch)
      epoch += 1