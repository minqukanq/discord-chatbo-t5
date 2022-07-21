import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from data_loader import QADataset
from utils import extract_question_and_answer

def train(epoch, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function
    """

    model.train()
    print_every = 100
    train_loss, train_num = 0,0
    for step, data in enumerate(loader, 1):
        ids = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)
        
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels
        )
        
        loss = outputs[0]
        
        train_loss += loss.item()
        train_num += ids.shape[0]
        if step % print_every == 0:
            print(f'epoch: {epoch}, instances: {train_num}, train loss: {round(train_loss/train_num,5)} ')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions
    """

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for step, data in enumerate(loader, 1):
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            answer = data['answer_text']
            
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=80, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                use_cache=True
            )
            
            pred = [
                tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for generated_id in generated_ids
            ]
            
            if step%10==0:
                print(f'log Completed {step}')
                
            predictions.append(pred)
            actuals.append(answer)

    return predictions, actuals

def T5Trainer(
    dataframe, model_params
):
    """
    T5 trainer
    """
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params.seed)  # pytorch random seed
    np.random.seed(model_params.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # logging
    print(f"""[Model]: Loading {model_params.model}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params.model)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = MT5ForConditionalGeneration.from_pretrained(model_params.model)
    if torch.cuda.device_count() > 1:
        print("Let's use", args.n_gpu, "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # logging
    print(f"[Data]: Reading data...\n")

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params.seed)
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f"FULL Dataset: {dataframe.shape}")
    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = QADataset(
        train_dataset,
        tokenizer,
        model_params.max_source_len,
        model_params.max_target_len,
    )

    val_set = QADataset(
        val_dataset,
        tokenizer,
        model_params.max_source_len,
        model_params.max_target_len,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params.train_batch_size,
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params.valid_batch_size,
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params.lr
    )

    # Training loop
    print(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params.epochs):
        train(epoch, model, device, training_loader, optimizer)

    print(f"[Saving Model]...\n")
    # Saving the model after training
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    path = os.path.join(args.output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    print(f"[Initiating Validation]...\n")
    for epoch in range(model_params.epochs):
        predictions, actuals = validate(tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(args.output_dir, "predictions.csv"))

    print(f"[Validation Completed.]\n")
    print(
        f"""[Model] Model saved @ {os.path.join(args.output_dir, "model_files")}\n"""
    )
    print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(args.output_dir,'predictions.csv')}\n"""
    )

def arg_parse():
    parser = argparse.ArgumentParser(description="This is a parser for train T5 model")
    parser.add_argument("--model", type=str, default="google/mt5-base", help="pretrained model")
    parser.add_argument("--data_path", type=str, required=True, help="data path")
    parser.add_argument("--train_batch_size", type=int, default=1, help="training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="validating batch size")
    parser.add_argument("--epochs", type=int, default=1, help="number of training epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--max_source_len", type=int, default=400, help="max length of source text")
    parser.add_argument("--max_target_len", type=int, default=32, help="max length of target text")
    parser.add_argument("--output_dir", type=str, default="./outputs/", help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="set seed for reproducibility")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = arg_parse()

    dataframe = extract_question_and_answer(args.data_path)
    T5Trainer(
        dataframe=dataframe,
        model_params=args
    )