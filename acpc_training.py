import torch
import time
import os
import multiprocessing
import traceback
from acpc_dataset import AcpcDataset, load_random_df
from poker_bert_models import BertPokerValueModel
from datetime import datetime
from transformers import BertTokenizer
from tqdm import tqdm
from loguru import logger
from statistics import mean

DATETIME_FORMAT = "%m-%d-%Y_%H:%M"

EPOCHS = 20
LEARNING_RATE = 1e-5
CLAMP_LOSS = False
CLAMP_LOSS_TH = 3
VALIDATION_DEVICE = 1

TRAIN_ROWS = 2*1024*1024
TEST_ROWS = 192*1024

# At least 512 to get gain from parallelization
BATCH_SIZE = 1024
MINI_BATCH_SIZE = 4
MINI_BATCH_VAL_SIZE = 4
MINI_BATCH_TEST_SIZE = 1

DO_TRAIN = True
MODEL_CHECKPOINT = "./models/bert_train_002m_val_0644.zip"
# MODEL_CHECKPOINT = "./models/bert_river_14m_val_0776.zip"
TRAIN_STREET = 3

PRINT_INFERENCE = True


def forward_pass(model, input_data, correct_label, device, mini_batch_size):
    correct_label = correct_label.to(device)
    correct_label = correct_label.unsqueeze(1)

    output = model(input_data, device)

    criterion = torch.nn.MSELoss(reduction = 'none')
    # criterion = torch.nn.HuberLoss(reduction = 'none')
    # criterion = torch.nn.BCELoss(reduction = 'none') # Binary cross entropy
    # criterion = torch.nn.HingeEmbeddingLoss(reduction = 'none')

    batch_loss = criterion(output, correct_label.float())

    # Clamp to -1 - 1
    if CLAMP_LOSS:
        batch_loss = torch.clamp(batch_loss, min=-CLAMP_LOSS_TH, max=CLAMP_LOSS_TH)

    # Average across mini batch
    batch_loss = batch_loss.sum() / mini_batch_size

    # Calc accuracy
    acc = (output * correct_label > 0).sum().item() / mini_batch_size
    return batch_loss, acc, output


def train_worker(result_dict, model, train_batch_df, device_id):

    try:
        torch.cuda.set_device(device_id)

        nrows = len(train_batch_df)
        train_dataset = AcpcDataset(train_batch_df, model)

        #logger.info(f"Running training on device {device_id}.")
        num_mini_batches = nrows // MINI_BATCH_SIZE

        start = time.time()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=MINI_BATCH_SIZE, shuffle=False
        )
        create_dataloader_time = time.time() - start

        acc_train = 0
        loss_train = 0

        start = time.time()
        model.to(device_id)
        copy_model_to_device_time = time.time() - start

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        start = time.time()

        for train_input, train_label, _ in train_dataloader:

            batch_loss, acc, _ = forward_pass(
                model, train_input, train_label, device_id, MINI_BATCH_SIZE
            )

            batch_loss = batch_loss / num_mini_batches

            loss_train += batch_loss.item()
            acc_train += (acc / num_mini_batches)

            batch_loss.backward()

        gradient_compute_time = time.time() - start
        
        start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_step_time = time.time() - start
        
        start = time.time()
        model.to(torch.device("cpu"))
        model_to_cpu_time = time.time() - start

        result_dict["model"] = model
        result_dict["acc_train"] = acc_train
        result_dict["loss_train"] = loss_train

        result_dict["create_dataloader_time"] = create_dataloader_time
        result_dict["copy_model_to_device_time"] = create_dataloader_time
        result_dict["gradient_compute_time"] = gradient_compute_time
        result_dict["optimizer_step_time"] = optimizer_step_time
        result_dict["model_to_cpu_time"] = model_to_cpu_time

    except Exception as e:
        logger.error(f"Training process crashed for device_id {device_id}")
        logger.error(traceback.format_exc())


def weight_normalizer(models) -> torch.nn.Module:

    state_dict_acc = models[0].state_dict()

    for key in state_dict_acc.keys():

        for model_x in models[1:]:
            state_dict_acc[key] = state_dict_acc[key] + model_x.state_dict()[key]

        state_dict_acc[key] = state_dict_acc[key] / len(models)

    models[0].load_state_dict(state_dict_acc)
    return models[0]


def train(model, train_dataset_path, val_dataset_path, epochs):

    val_df = load_random_df(val_dataset_path, TEST_ROWS, TRAIN_STREET)
    val_dataset = AcpcDataset(val_df, model)

    num_cuda_devices = torch.cuda.device_count()

    for epoch_num in range(epochs):
        
        logger.info(f"")
        logger.info(f"Starting epoch {epoch_num}. Training street {TRAIN_STREET}")
        logger.info(f"Loading random train rows {TRAIN_ROWS}")
        logger.info(f"Batch size {BATCH_SIZE}")
        logger.info(f"Validation rows {TEST_ROWS}")
        logger.info(f"Num CUDA devices {num_cuda_devices}")

        train_df = load_random_df(train_dataset_path, TRAIN_ROWS, TRAIN_STREET)
        num_batches = TRAIN_ROWS // BATCH_SIZE

        total_loss_train = []
        total_acc_train = []

        create_dataloader_time = []
        copy_model_to_device_time = []
        gradient_compute_time = []
        optimizer_step_time = []
        model_to_cpu_time = []
        weight_normalizer_time = []

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
                    train_batch_df = train_df[skip_rows:skip_rows+nrows]

                    process = multiprocessing.Process(
                        target=train_worker,
                        args=[
                            result_dict,
                            model,
                            train_batch_df,
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
                    
                    if "model" in result_dict and "model_to_cpu_time" in result_dict:
                        models.append(result_dict["model"])
                        total_loss_train.append(result_dict["loss_train"])
                        total_acc_train.append(result_dict["acc_train"])

                        create_dataloader_time.append(result_dict["create_dataloader_time"])
                        copy_model_to_device_time.append(result_dict["copy_model_to_device_time"])
                        gradient_compute_time.append(result_dict["gradient_compute_time"])
                        optimizer_step_time.append(result_dict["optimizer_step_time"])
                        model_to_cpu_time.append(result_dict["model_to_cpu_time"])

                # Average models
                #logger.info(f"Averaging models from different processes")
                start = time.time()

                # Update models only if some models are back
                if len(models) > 1:
                    model = weight_normalizer(models)
                elif len(model) == 1:
                    model = models[0]
                
                weight_normalizer_time.append(time.time() - start)

        logger.info(f"create_dataloader_time: {mean(create_dataloader_time):.3f}s")
        logger.info(f"copy_model_to_device_time: {mean(copy_model_to_device_time):.3f}s")
        logger.info(f"gradient_compute_time: {mean(gradient_compute_time):.3f}s")
        logger.info(f"optimizer_step_time: {mean(optimizer_step_time):.3f}s")
        logger.info(f"model_to_cpu_time: {mean(model_to_cpu_time):.3f}s")
        logger.info(f"weight_normalizer_time: {mean(weight_normalizer_time):.3f}s")

        # Save model
        date = datetime.now().strftime(DATETIME_FORMAT)
        model_name = f"./models/bert_{date}.zip"

        logger.info(f"Saving model to {model_name}")
        torch.save(model.state_dict(), model_name)

        # Copy model to cuda device for validation
        model.to(VALIDATION_DEVICE)

        # Run validation
        logger.info(f"Running validation")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=MINI_BATCH_VAL_SIZE)

        total_loss_val = []
        total_acc_val = []

        with torch.no_grad():
            for val_input, val_label, _ in tqdm(val_dataloader):

                batch_loss, acc, _ = forward_pass(
                    model, val_input, val_label, VALIDATION_DEVICE, MINI_BATCH_VAL_SIZE
                )

                total_loss_val.append(batch_loss.item())
                total_acc_val.append(acc)

        logger.info(f"Epoch {epoch_num} result: ")
        logger.info(f"Train Loss: {mean(total_loss_train):.3f}")
        logger.info(f"Train Accuracy: {mean(total_acc_train):.3f}")
        logger.info(f"Val Loss: {mean(total_loss_val): .3f}")
        logger.info(f"Val Accuracy: {mean(total_acc_val): .3f}")

        # After validation is done copy model back to CPU 
        model.to(torch.device("cpu"))


def evaluate(model, test, print_inference=True):

    device = torch.device(f"cuda:{VALIDATION_DEVICE}")
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=MINI_BATCH_TEST_SIZE)

    model = model.to(device)
    total_acc_test = []
    total_loss_test = []

    with torch.no_grad():
        for test_input, test_label, train_text in tqdm(test_dataloader):

            batch_loss, acc, out_label = forward_pass(model, test_input, test_label, device, MINI_BATCH_TEST_SIZE)
            total_acc_test.append(acc)
            total_loss_test.append(batch_loss.item())

            if print_inference:
                run_inference_result = model.run_inference(train_text, device)

                logger.info(f"Input: {train_text}")
                logger.info(f"correct_label {test_label.item()} -> predicted label {out_label.item(): .3f} : run_inference_result {run_inference_result: .3f}, loss {batch_loss.item(): .3f}, acc {acc: .3f} ")

    logger.info(f"Test Accuracy: {mean(total_acc_test): .3f}")
    logger.info(f"Test Loss: {mean(total_loss_test): .3f}")


if __name__ == "__main__":
    
    date = datetime.now().strftime(DATETIME_FORMAT)
    logger.add(f"./logs/log_{date}.txt")

    # Set start method to spawn to use cuda
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()

    logger.info(f"Cuda is available {torch.cuda.is_available()}")
    logger.info(f"Dev count {torch.cuda.device_count()}")

    # Train the model
    model = BertPokerValueModel(TRAIN_STREET) 
    model.load_from_checkpoint(MODEL_CHECKPOINT)

    if DO_TRAIN:
        logger.info("Started training process")
        train(model, "data/acpc_train.txt", "data/acpc_val.txt", EPOCHS)

    # Evaluate model
    test_df = load_random_df("data/acpc_test.txt", TEST_ROWS, TRAIN_STREET)
    test = AcpcDataset(test_df, model)
    evaluate(model, test, print_inference=PRINT_INFERENCE)
