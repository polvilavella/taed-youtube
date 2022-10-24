---
language: english
datasets:
- Youtube Statistics
authors: Pol Vilavella, Pol Costa, Daniel Gonzálbez, Eva Brigitte
---
# Model Card for DistilBERTforSentiment 

## Model description

DistilBERTforSentiment is a fine-tuned version of the DistilBERT model (see details in: https://huggingface.co/distilbert-base-cased), which was pretrained on the same corpus as BERT  model (which worked as a teacher during training of DistilBERT) in a self-supervised way. This model is faster and computationally much more efficient than the BERT model.
Apart from DistilBERT (which outputs a result of dimension 768), our model also contains two more blocks: an adaptor with 2 linear layers (with a hidden size of 32) and an activation function (ReLU) with a dropout of 0.2 and a classifier, which decides whether the input comment is positive or negative. DistilBERTforSentiment has been trained in a self-supervised manner. 


## Intended uses

The model is used to process a comment and classify it into positive or negative, depending on how it has been written.


## Limitations and biases

The training data used to train this model could be biased. As classifying a text into a sentiment is totally subjective, the data is biased by the dataset author's perspective. Also, the DistilBERT model inherits the biases of the BERT model, as it was trained to imitate its behaviour.

The main limitation our model faces is the fact that a lot of youtube comments may contain unknown words (because the way the users write could be grammatically incorrect) for the DistilBERT model that, when tokenizing them, will be classified as unknown. 

## Recommendations

Users must be aware of the previously mentioned limitations, as they should not be using this model for some kind of comments, like the ones that entirely express their sentiment with not well-written words (like with abbreviations or acronyms). The comments should be entirely written in English as the model does not support other languages.


## How to use the model

Here is how to use this model to get the features of a given text in PyTorch...

```python
from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertModel.from_pretrained("distilbert-base-cased")
comment = "Good video, my friend."
encoded_input = tokenizer(comment, padding=True, truncation = True, max_length=512,return_tensors="pt")['input_ids']
output = model(encoded_input)
```
... and the output should be something like: [-0.15, 1.27]


## Training data

DistilBERTforSentiment was trained on the Youtube-Statistics dataset, specifically on the comments.csv file selecting the comments that correspond to positive and negative english comments.


## Training procedure

### Preprocessing

As in DistilBERT, the given comments are firstly tokenized using WordPiece and a vocabulary size of 30,000. The target for each example in the dataset is converted from an integer (0 or 2) to a tensor ([1,0], [0,1]). 

### Training
See the [training code](https://https://github.com/polvilavella/taed-youtube) for all hyperparameters details.

