
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm

from dataloader import DataLoader
from model import Model

class Train():

    def __init__(self, 
                 config : dict, 
                 vocab_dict : dict,
                 ):
        
        self.config = config

        self.assertConfig()

        self.model = Model(vocab_dict)

        self.inputs, self.labels = DataLoader(config["traindata_path"], vocab_dict).loadTrain()
        self.inputs_test, self.labels_test = DataLoader(config["testdata_path"], vocab_dict).loadVal()
        self.inputs_val, self.labels_val = DataLoader(config["valdata_path"], vocab_dict).loadVal()

        wandb.init(project=config["project"], name=config["run_name"])
        wandb.config.update(config)

    def assertConfig(self)->None:

        assert "project" in self.config, "<class Train>: project is not in config"
        assert "run_name" in self.config, "<class Train>: run_name is not in config"
        assert "epochs" in self.config, "<class Train>: epochs is not in config"
        assert "lr" in self.config, "<class Train>: lr is not in config"
        assert "patience" in self.config, "<class Train>: patience is not in config"
        

    def test(self) -> None:
        
        check = 0
        count = 0
        with torch.no_grad():
            for input_item, label_item in zip(self.inputs_test, self.labels_test):
                input_item = input_item.to(self.model.device)
                label_item = label_item.to(self.model.device)
                output = self.model(input_item)
                F.softmax(output, dim=-1)
                model_choice = torch.argmax(output, dim=-1)
                if torch.equal(model_choice, label_item):
                    check += 1
                count += 1
        wandb.log({"test_acc1": check/count})

    def validate(self) -> float:
        
        check = 0
        count = 0
        val_loss = 0
        with torch.no_grad():
            for input_item, label_item in zip(self.inputs_val, self.labels_val):
                input_item = input_item.to(self.model.device)
                label_item = label_item.to(self.model.device)
                output = self.model(input_item)
                F.softmax(output, dim=-1)
                val_loss += self.criterion(output, label_item).item()
                model_choice = torch.argmax(output, dim=-1)
                if torch.equal(model_choice, label_item):
                    check += 1
                count += 1
        
        val_loss /= count
        wandb.log({"val_loss": val_loss, 
                   "val_acc1": check/count})
        
        return val_loss

    def train(self) -> None:

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # Early stopping parameters
        patience = self.config["patience"]
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # training loop
        for epoch in tqdm.tqdm(range(self.config["epochs"])): 

            # train from training data
            for input_batch, label_batch in zip(self.inputs, self.labels):
                input_batch = input_batch.to(self.model.device)
                label_batch = label_batch.to(self.model.device)

                optimizer.zero_grad()
                output = self.model(input_batch)
                loss = self.criterion(output, label_batch)
                loss.backward()
                optimizer.step()

                # log
                wandb.log({"loss": loss.item(), 
                           "epoch": epoch})

            # test and validation after each epoch(freeze the model)
            self.model.eval()

            # test
            self.test()

            # validation
            val_loss = self.validate()
            
            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print("Early stopping at epoch {}".format(epoch))
                break
            
            # save model
            torch.save(self.model.state_dict(), "./model/checkpoints/model_{}.pt".format(self.config["run_name"]))
            wandb.save("./model/checkpoints/model_{}.pt".format(self.config["run_name"]))