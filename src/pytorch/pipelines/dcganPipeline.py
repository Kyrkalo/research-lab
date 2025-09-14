from src.pytorch.trainers.dcganTrainers import DCGANTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch.models.dcgan import GanDiscriminator, GanGenerator, weights_init
from pytorch.dataloaders.dcganDataLoader import DcganDataLoader

class DcganPipeline:
    
    def __init__(self, configs=None):
        self.device = configs["device"] if configs and "device" in configs else self.device
        self.dcgan_config = configs if configs else self.dcgan_config
        pass

    def setup(self):
        
        self.modelG = GanGenerator(self.dcgan_config).to(self.device)
        self.modelD = GanDiscriminator(self.dcgan_config).to(self.device)

        self.modelG.apply(weights_init)
        self.modelD.apply(weights_init)

        self.optimizerG = optim.Adam(self.modelG.parameters(), lr=self.dcgan_config["lr"], betas=(self.dcgan_config["beta1"], 0.999))
        self.optimizerD = optim.Adam(self.modelD.parameters(), lr=self.dcgan_config["lr"], betas=(self.dcgan_config["beta1"], 0.999))

        self.criterion = nn.BCELoss()

        self.dataloader = DcganDataLoader(self.dcgan_config).Get()

        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.dcgan_config["random_seed"])        

        self.trainer = DCGANTrainer(self.modelG, self.modelD, self.optimizerG, self.optimizerD, self.criterion, self.dataloader, self.device, self.dcgan_config)
        return self

    def run(self):
        for epoch in range(1, self.dcgan_config["num_epochs"] + 1):
            self.trainer.train(epoch)
        return self    