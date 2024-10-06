import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabdict import VocabDict
from dataloader import DataLoader
from model import Model
import wandb
import tqdm
import os

DEV_PATH = "./data/dev.txt"
TRAIN_PATH = "./data/train.txt"
TEST_PATH = "./data/test.txt"
VOCABDICT_PATH = "./model/vocab/vocabset.csv"

if __name__ == "__main__":

    vocab_dict = VocabDict.loadVocabDict(VOCABDICT_PATH)
    
    inputs, labels = DataLoader(TRAIN_PATH, vocab_dict).loadTrain()
    model = Model(vocab_dict)

    a = input("Press Enter to start training...")

    wandb.init(project="nlp_homework1")

    wandb.config.update({
        "batch_size": 64,
        "timeseq_len": 32,
        "learning_rate": 0.001,
        "embedding_dim": 128,
        "class_num": 4,
        "filter_windows": [3, 4, 5],
        "filter_num": 100,
        "dropout_p": 0.5,
        "l2_constraint": 3.0,
        "linear_hidden_layers": [512, 64],
        "device": "auto"
    })

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(range(100)): 
        for input_batch, label_batch in zip(inputs, labels):
            input_batch = input_batch.to(model.device)
            label_batch = label_batch.to(model.device)

            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

        # validation
        inputs_val, labels_val = DataLoader(DEV_PATH, vocab_dict).loadVal()
        check = 0
        count = 0
        for input_item, label_item in zip(inputs_val, labels_val):
            input_item = input_item.to(model.device)
            label_item = label_item.to(model.device)
            output = model(input_item)
            F.softmax(output, dim=-1)
            model_choice = torch.argmax(output, dim=-1)
            if torch.equal(model_choice, label_item):
                check += 1
            count += 1
        
        wandb.log({"val_acc1": check/count})
        wandb.log({"epoch": epoch})
            
        torch.save(model.state_dict(), f"./model/checkpoints/model.pt")
        wandb.save(f"./model/checkpoints/model.pt")

    