import os
import torch
from src.notebook.engine import evaluate, train_one_epoch

class RCnnTrainer:
    def __init__(self, model, data_loader, data_test, config):
        self.config = config
        self.device = config["device"]
        self.data_loader = data_loader
        self.data_test = data_test
        self.model = model

    def train(self):
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.config["learning_rate"],
            momentum=0.9,
            weight_decay=0.0005
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )
        for epoch in range(self.config["num_epochs"]):
            
            train_one_epoch(self.model, optimizer, self.data_loader, self.device, epoch, print_freq=100)
            lr_scheduler.step()
            evaluate(self.model, self.data_test, device=self.device)
            path = os.path.join(self.config["model_name"] + ".pth")
            torch.save(self.model.state_dict(), path)
        pass