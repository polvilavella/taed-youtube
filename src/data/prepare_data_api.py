
""" Module to prepare the data for the API calls. """

import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer


MODEL = "distilbert-base-cased"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL, max_length=512, padding=True,
                                                truncation = True, return_tensors="pt")


class DistilBERTforSentiment(nn.Module):
    """ Neural Network definition. """
    def __init__(self, adapter_hidden_size=32):
        super().__init__()

        self.distilbert = DistilBertModel.from_pretrained(MODEL)

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
        """ Forward step of the model. """
        outputs = self.distilbert(input_ids = inputs, return_dict=False)
        # B x seq_length x H
        x = self.adaptor(outputs[0])

        x,_ = x.max(dim=1)
        # B x H

        results = self.classifier(x)
        return results


def model():
    """ Function to get the model. """
    model = DistilBERTforSentiment(adapter_hidden_size=32)
    return model



class Loader(torch.utils.data.Dataset):
    """ Data preparation. """
    def __init__(self, comments):
        self.data = tokenizer(comments, padding=True, truncation = True,
                              max_length=512,return_tensors="pt")['input_ids']

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)


def prepare_comment(comment):
    """ Prepare the comment text to make the prediction. """
    pre_loader = Loader(comment)
    loader = torch.utils.data.DataLoader(pre_loader, batch_size=1,
                                          shuffle=True, num_workers=0)
    return loader
