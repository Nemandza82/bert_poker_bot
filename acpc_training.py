import torch
import math
import time
import numpy as np
from acpc_dataset import AcpcDataset
from poker_bert_models import BertPokerValueModel
from datetime import datetime
from torch.optim import Adam
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP


EPOCHS = 20
LR = 1e-6
TRAIN_ROWS = 10000000 # 10 miliona
TEST_ROWS = 500000


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


#def train_step(rank, world_size):


def train(model, train_dataset_path, val_dataset_path, learning_rate, epochs, use_cuda, device):

    train_dataset = AcpcDataset(train_dataset_path, 0, TRAIN_ROWS, model.get_tokenizer())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dataset = AcpcDataset(val_dataset_path, 0, TEST_ROWS, model.get_tokenizer())
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        #model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                train_label = train_label.unsqueeze(1)

                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.float())
                total_loss_train += batch_loss.item()

                acc = (output * train_label > 0).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # Save model
            date = datetime.now().strftime(datetime_format)
            model_name = f"./models/bert_{date}.zip"

            print(f"Saving model to {model_name}")
            torch.save(model.state_dict(), model_name)

            total_loss_val = 0
            total_acc_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    val_label = val_label.unsqueeze(1)

                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.float())
                    total_loss_val += batch_loss.item()

                    acc = (output * val_label > 0).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataset): .3f}')


def evaluate(model, test, use_cuda, device):

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    with torch.no_grad():
        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              test_label = test_label.unsqueeze(1)

              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output * test_label > 0).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


print(f"Cuda is available {torch.cuda.is_available()}")
print(f"Dev count {torch.cuda.device_count()}")

datetime_format = "%m-%d-%Y_%H:%M"
use_cuda = torch.cuda.is_available()

if use_cuda:
    print("Using CUDA!")
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
else:
    print("Not using CUDA!")
    device = torch.device("cpu")

# Train the model
model = BertPokerValueModel()
model.load_from_checkpoint("./models/bert_train_069_val_061.zip")

do_train = True

if do_train:

    print("Started training process")
    train(model, "data/acpc_train.txt", "data/acpc_val.txt", LR, EPOCHS, use_cuda, device)

    # Save model
    date = datetime.now().strftime(datetime_format)
    model_name = f"./models/bert_{date}.zip"

    print(f"Saving model to {model_name}")
    torch.save(model.state_dict(), model_name)

# Evaluate model
test = AcpcDataset("data/acpc_test.txt", 0, TEST_ROWS, model.get_tokenizer())
evaluate(model, test, use_cuda, device)
