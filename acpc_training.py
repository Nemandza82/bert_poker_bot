import torch
import time
import os
from acpc_dataset import AcpcDataset
from poker_bert_models import BertPokerValueModel
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from multiprocessing import freeze_support
from transformers import BertTokenizer


EPOCHS = 20
LEARNING_RATE = 1e-6
#TRAIN_ROWS = 10000000 # 10 miliona
#TEST_ROWS = 500000

TRAIN_ROWS = 200
TEST_ROWS = 40


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def forward_pass(model, input_data, correct_label, criterion, device):
    correct_label = correct_label.to(device)
    correct_label = correct_label.unsqueeze(1)

    mask = input_data["attention_mask"].to(device)
    input_id = input_data["input_ids"].squeeze(1).to(device)

    output = model(input_id, mask)

    if criterion is not None:
        batch_loss = criterion(output, correct_label.float())
    else:
        batch_loss = 0

    acc = (output * correct_label > 0).sum().item()
    return batch_loss, acc


def train_step(rank, world_size, train_dataset, model):

    batch_size = 1
    nsteps = (TRAIN_ROWS // world_size) // batch_size
    skip_steps = rank * nsteps

    torch.cuda.set_device(rank)
    cpu_device = torch.device("cpu")

    print(f"Running training on rank {rank}. nstep {nsteps} skiprows {skip_steps}")

    if world_size > 1:
        setup(rank, world_size)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )

    total_acc_train = 0
    total_loss_train = 0

    model.to(rank)

    if world_size > 1 and False:
        print(f"Create DDP for rank {rank}")
        ddp_model = DistributedDataParallel(model, device_ids=[rank], gradient_as_bucket_view=True)
        print(f"Created DDP for rank {rank}")
    else:
        ddp_model = model

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=LEARNING_RATE)
    step = 0

    for train_input, train_label in train_dataloader:

        if step >= skip_steps and step < skip_steps + nsteps:

            print(f"Rank {rank} step {step} / {len(train_dataloader)}")

            batch_loss, acc = forward_pass(
                ddp_model, train_input, train_label, criterion, rank
            )
            total_loss_train += batch_loss.item()
            total_acc_train += acc

            ddp_model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        step += 1

    if world_size > 1:
        cleanup()


def train(model, train_dataset_path, val_dataset_path, epochs):

    val_dataset = AcpcDataset(val_dataset_path, 0, TEST_ROWS, tokenizer)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

    world_size = torch.cuda.device_count()

    if world_size == 1:
        device = torch.device("cuda:0")

    train_dataset = AcpcDataset(
        train_dataset_path, 0, TRAIN_ROWS, tokenizer
    )

    for epoch_num in range(epochs):

        if world_size > 1:
            torch.multiprocessing.spawn(
                train_step,
                args=(world_size, train_dataset, model),
                nprocs=world_size,
                join=True,
            )
        else:
            train_step(0, 1, train_dataset, model)

        # Save model
        date = datetime.now().strftime(datetime_format)
        model_name = f"./models/bert_{date}.zip"

        print(f"Saving model to {model_name}")
        torch.save(model.state_dict(), model_name)

        model.to(device)
        criterion = torch.nn.MSELoss()

        total_loss_val = 0
        total_acc_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:

                batch_loss, acc = forward_pass(
                    model, val_input, val_label, criterion, device
                )

                total_loss_val += batch_loss.item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
            | Val Loss: {total_loss_val / len(val_dataset): .3f} \
            | Val Accuracy: {total_acc_val / len(val_dataset): .3f}"
        )


def evaluate(model, test):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    model = model.to(device)
    total_acc_test = 0

    with torch.no_grad():
        for test_input, test_label in test_dataloader:

            batch_loss, acc = forward_pass(model, test_input, test_label, None, device)

            acc = (output * test_label > 0).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


if __name__ == "__main__":
    freeze_support()

    print(f"Cuda is available {torch.cuda.is_available()}")
    print(f"Dev count {torch.cuda.device_count()}")

    datetime_format = "%m-%d-%Y_%H:%M"

    # Train the model
    model = BertPokerValueModel()
    model.load_from_checkpoint("./models/bert_train_069_val_061.zip")

    do_train = True

    if do_train:
        print("Started training process")
        train(model, "data/acpc_train.txt", "data/acpc_val.txt", EPOCHS)

    # Evaluate model
    test = AcpcDataset("data/acpc_test.txt", 0, TEST_ROWS, tokenizer)
    evaluate(model, test)
