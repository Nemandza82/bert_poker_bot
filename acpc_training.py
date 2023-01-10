import torch
import time
import os
import multiprocessing
from acpc_dataset import AcpcDataset
from poker_bert_models import BertPokerValueModel
from datetime import datetime
from transformers import BertTokenizer
from tqdm import tqdm
from loguru import logger

DATETIME_FORMAT = "%m-%d-%Y_%H:%M"

EPOCHS = 20
LEARNING_RATE = 1e-6

#F = 10000000 # 10 miliona
#TEST_ROWS = 500000

#TRAIN_ROWS = 256
#TEST_ROWS = 64

TRAIN_ROWS = 256000
TEST_ROWS = 6400

BATCH_SIZE = 128


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


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


def train_worker(result_dict, model, train_dataset_path, skip_rows, nrows, device_id):

    torch.cuda.set_device(device_id)

    #logger.info(f"Running training on device {device_id}.")

    train_dataset = AcpcDataset(
        train_dataset_path, skip_rows, nrows, tokenizer
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False
    )

    acc_train = 0
    loss_train = 0

    model.to(device_id)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for train_input, train_label in train_dataloader:

        batch_loss, acc = forward_pass(
            model, train_input, train_label, criterion, device_id
        )
        loss_train += batch_loss.item()
        acc_train += acc

        model.zero_grad()
        batch_loss.backward()
    
    optimizer.step()
    model.to(torch.device("cpu"))

    result_dict["model"] = model
    result_dict["acc_train"] = acc_train
    result_dict["loss_train"] = loss_train


def weight_normalizer(models) -> torch.nn.Module:

    state_dict_acc = models[0].state_dict()

    for key in state_dict_acc.keys():

        for model_x in models[1:]:
            state_dict_acc[key] = state_dict_acc[key] + model_x.state_dict()[key]

        state_dict_acc[key] = state_dict_acc[key] / len(models)

    models[0].load_state_dict(state_dict_acc)
    return models[0]


def train(model, train_dataset_path, val_dataset_path, epochs):

    val_dataset = AcpcDataset(val_dataset_path, 0, TEST_ROWS, tokenizer)
    num_cuda_devices = torch.cuda.device_count()

    for epoch_num in range(epochs):
        
        logger.info(f"")
        logger.info(f"Starting epoch {epoch_num}")
        logger.info(f"Train rows {TRAIN_ROWS}")
        logger.info(f"Batch size {BATCH_SIZE}")
        logger.info(f"Num CUDA devices {num_cuda_devices}")

        num_batches = TRAIN_ROWS // BATCH_SIZE

        total_loss_train = 0
        total_acc_train = 0

        # Train in batch in parallel on multiple GPUs
        for batch_id in tqdm(range(num_batches)):
            
            with multiprocessing.Manager() as manager:

                # Store each process in this dict
                processes = {}

                # Each cuda device is processing nrows
                nrows = BATCH_SIZE // num_cuda_devices

                # Create process for each cuda device
                for device_id in range(num_cuda_devices):

                    # Create manager dict for passing back parameters from process
                    result_dict = manager.dict()

                    skip_rows = batch_id * BATCH_SIZE + device_id * nrows

                    process = multiprocessing.Process(
                        target=train_worker,
                        args=[
                            result_dict,
                            model,
                            train_dataset_path,
                            skip_rows,
                            nrows,
                            device_id
                        ],
                    )

                    process.start()
                    processes[device_id] = (process, result_dict)

                models = []

                # Join processes
                for device_id, (process, result_dict) in processes.items():
                    #logger.info(f"Joining process {device_id}")
                    process.join()
                    
                    models.append(result_dict["model"])
                    total_loss_train += result_dict["loss_train"]
                    total_acc_train += result_dict["acc_train"]

                # Average models
                #logger.info(f"Averaging models from different processes")
                model = weight_normalizer(models)

        avg_loss_train = total_loss_train / (num_batches * BATCH_SIZE)
        avg_acc_train = total_acc_train / (num_batches * BATCH_SIZE)

        # Save model
        date = datetime.now().strftime(DATETIME_FORMAT)
        model_name = f"./models/bert_{date}.zip"

        logger.info(f"Saving model to {model_name}")
        torch.save(model.state_dict(), model_name)

        # Copy model to cuda device for validation
        validation_dev = 0
        model.to(validation_dev)

        # Run validation
        logger.info(f"Running validation")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
        criterion = torch.nn.MSELoss()

        total_loss_val = 0
        total_acc_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:

                batch_loss, acc = forward_pass(
                    model, val_input, val_label, criterion, validation_dev
                )

                total_loss_val += batch_loss.item()
                total_acc_val += acc

        logger.info(f"Epochs: {epoch_num + 1}")
        logger.info(f"Train Loss: {avg_loss_train:.3f}")
        logger.info(f"Train Accuracy: {avg_acc_train:.3f}")
        logger.info(f"Val Loss: {total_loss_val / len(val_dataset): .3f}")
        logger.info(f"Val Accuracy: {total_acc_val / len(val_dataset): .3f}")

        # After validation is done copy model back to CPU 
        model.to(torch.device("cpu"))


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

    logger.info(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


if __name__ == "__main__":
    
    date = datetime.now().strftime(DATETIME_FORMAT)
    logger.add(f"./logs/log_{date}.txt")

    # Set start method to spawn to use cuda
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()

    logger.info(f"Cuda is available {torch.cuda.is_available()}")
    logger.info(f"Dev count {torch.cuda.device_count()}")

    # Train the model
    model = BertPokerValueModel()
    model.load_from_checkpoint("./models/bert_train_069_val_061.zip")

    do_train = True

    if do_train:
        logger.info("Started training process")
        train(model, "data/acpc_train.txt", "data/acpc_val.txt", EPOCHS)

    # Evaluate model
    test = AcpcDataset("data/acpc_test.txt", 0, TEST_ROWS, tokenizer)
    evaluate(model, test)
