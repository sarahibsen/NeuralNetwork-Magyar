# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:41:56 2024

@author: ibsens

Train Neural Translation Model
"""
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:41:56 2024

@author: ibsens

Train Neural Translation Model
"""

import torch
import torch.nn as nn
from pickle import load 
from torchtext.data.utils import get_tokenizer
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)

# load clean data set 
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# load them data sets
dataset = load_clean_sentences('english-magyar-both.pkl')
train = load_clean_sentences('english-magyar-train.pkl')
test = load_clean_sentences('english-magyar-test.pkl')

print(dataset[0][0])
# tokenize data first 
# TODO: tokenize
def tokenize_dataset(dataset, text_transform):
    tokenized_data = []
    for item in dataset:
        if len(item) < 2:
            print(f"Not enough data to unpack for item: {item}")
            continue
        if len(item) > 2:
            print(f"Extra data in item, handling separately: {item[2]}")
        english, magyar = item[:2]
        tokenized_english = text_transform(english)
        tokenized_magyar = text_transform(magyar)
        tokenized_data.append((tokenized_english, tokenized_magyar))
    return tokenized_data


# tokenize all datasets
tokenized_dataset = tokenize_dataset(dataset, text_transform)
tokenized_train = tokenize_dataset(train, text_transform)
tokenized_test = tokenize_dataset(test, text_transform)


#tokenizer = get_tokenizer("basic_english")
#TODO: loop through whole data set 

print(tokenized_dataset[0]) 



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 64), # token is the tokenization size of the input sentence, and ization is the expected output size
            #non linearities
            nn.ReLU(),
            nn.Linear(64,128),     # expects an input 
            nn.ReLU(),
            nn.Linear(128,5),            #and the output
            )
        
    def forward(self, x):
       # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#TODO: set up a simple training / test eval to understand how it works   
model = NeuralNetwork().to("cpu")
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()



def train_one_epoch(model, training_loader, loss_fn, optimizer, device="cpu"):
    model.train()  # Set the model to training mode
    total_loss = 0

    for inputs, labels in training_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(training_loader)
    print(f"Average loss: {average_loss:.4f}")
    return average_loss





