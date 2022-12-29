import torch
import math
import time
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    #def classes(self):
    #    return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(torch.nn.Module):
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_dataset, val_dataset, learning_rate, epochs, use_cuda, device):

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataset): .3f}')


def evaluate(model, test_data, use_cuda, device):

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    with torch.no_grad():
        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


print(f"Cuda is available {torch.cuda.is_available()}")
print(f"Dev count {torch.cuda.device_count()}")

datetime_format = "%m-%d-%Y_%H:%M"
use_cuda = torch.cuda.is_available()
#use_cuda = True
#use_cuda = False

if use_cuda:
    print("Using CUDA!")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    print("Not using CUDA!")
    device = torch.device("cpu")


# Read the dataset
datapath = 'bbc-text.csv'
df = pd.read_csv(datapath)
print(df.head())

# Tokenize example
example_text = 'I will watch Memento tonight'
bert_input = tokenizer(example_text,padding='max_length', max_length = 10, 
                       truncation=True, return_tensors="pt")

print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])

example_text = tokenizer.decode(bert_input.input_ids[0])
print(example_text)

# Split dataset
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

print(f"Traninig: {len(df_train)}, val: {len(df_val)}, test:{len(df_test)}")

# Train the model
EPOCHS = 1
LR = 1e-6
model = BertClassifier()

print(f"Loading model from ./models/bert_12-04-2022_18:23.zip")
model.load_state_dict(torch.load("./models/bert_12-04-2022_18:23.zip"))
model.eval()

do_train = True

if do_train:
    train_dataset, val_dataset = Dataset(df_train), Dataset(df_val)
    train(model, train_dataset, val_dataset, LR, EPOCHS, use_cuda, device)

    # Save model
    date = datetime.now().strftime(datetime_format)
    model_name = f"./models/bert_{date}.zip"

    print(f"Saving model to {model_name}")
    torch.save(model.state_dict(), model_name)

# Evaluate model
evaluate(model, df_test, use_cuda, device)