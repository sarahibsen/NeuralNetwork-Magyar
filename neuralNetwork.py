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


import csv
import torch
import wandb
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torchtext.transforms as T
from transformers import pipeline
from torch.nn.utils.rnn import pad_sequence
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader, Dataset

run = wandb.init(
    project = "magyar-english-translation",
    notes = "LLM",
    tags = ["baseline"],
    entity = "axient-ml"
    )

# moving operations to the gpu 
if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 
a = torch.zeros(4,3) 
a = a.to(device)

#data = pd.read_csv("./english-magyar.csv", engine='python', encoding = "UTF-8", sep =',\t', quotechar= '"')

def parse_complex_csv(file_path):
    english = []
    magyar = []

    with open(file_path, 'r', encoding='UTF-8') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        next(reader)  # Assuming there's a header to skip
        for row in reader:
            if len(row) != 2:
                # Attempt to intelligently join parts back together
                combined_english = ','.join(row[:-1]).strip('"')
                combined_magyar = row[-1].strip('"')
                english.append(combined_english)
                magyar.append(combined_magyar)
            else:
                english.append(row[0].strip('"'))
                magyar.append(row[1].strip('"'))

    return pd.DataFrame({
        'english': english,
        'magyar': magyar
    })

# Call the function with your file path
data = parse_complex_csv("./english-magyar.csv")

# Display the first few rows to confirm correct parsing
print(data.head())


def tokenize(sentence):
    return [word for word in sentence.split()]

# Apply tokenization
data['english_tokens'] = data['english'].apply(tokenize)
data['magyar_tokens'] = data['magyar'].apply(tokenize)

# initialize tensors on to the GPU 

X = torch.randn(len(data), 5, device = device)  # Example tensor, replace 5 with actual input size
y = torch.randint(0, 2, (len(data),), device = device)  # Example tensor, replace with actual target tensor

# Split dataset into training and testing sets
split_idx = int(0.8 * len(data))
Xtrain, Xtest = X[:split_idx], X[split_idx:]
ytrain, ytest = y[:split_idx], y[split_idx:]


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
        # literal flowingy of data from the beginning to the end
        return logits
    
#TODO: set up a simple training / test eval to understand how it works   
model = NeuralNetwork()
model.to(torch.device("cuda:0"))


print(model)

#loss function and optimizer 

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

# declare the number of epochs and batch size
n_epochs = 50 
batch_size = 10


for epoch in range(n_epochs):
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0

    for i in range(0, len(Xtrain), batch_size):
        Xbatch = Xtrain[i:i+batch_size]
        ybatch = ytrain[i:i+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # wandb.log({"batch_loss": loss.item()}, step=epoch * len(Xtrain) // batch_size + i // batch_size)
       # Calculate batch loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(y_pred.data, 1)
        total += ybatch.size(0)
        correct += (predicted == ybatch).sum().item()

    # Calculate average loss and accuracy
    train_loss /= len(Xtrain) // batch_size
    train_acc = 100 * correct / total

     
# evaluating the test dataset
    model.eval()
    # attempting to log all the metrics onto wandb
    test_loss = 0
    correct = 0
    total = 0 
    with torch.no_grad():
        for i in range(0, len(Xtest), batch_size):
            Xbatch = Xtest[i:i+batch_size]
            ybatch = ytest[i:i+batch_size]
            y_pred = model(Xbatch)
            
            # Calculate loss for the batch
            loss = loss_fn(y_pred, ybatch)
            test_loss += loss.item()
            
            # Calculate accuracy for the batch
            _, predicted = torch.max(y_pred.data, 1)
            total += ybatch.size(0)
            correct += (predicted == ybatch).sum().item()

    # Calculate average loss and accuracy for all batches
    test_loss /= len(Xtest) // batch_size
    test_acc = 100 * correct / total
        
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })
    print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
# Initializing in a separate cell so we can easily add more epochs to the same run
#timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

run.finish()


# features are input labels are the idealized output
# features are what we want the NN to see + learn from 
# labels are what we use to check the output

# features + labels are used for training 

