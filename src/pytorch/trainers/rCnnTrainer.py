import torch
from src.notebook.engine import train_one_epoch
from src.pytorch.helpers.coco_eval import evaluate

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
            lr=self.config["learlearning_rate"],
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
        pass