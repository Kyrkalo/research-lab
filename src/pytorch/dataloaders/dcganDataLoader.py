import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

class DcganDataLoader:
    def __init__(self, configs):
        self.dataroot = configs["dataroot"]
        self.image_size = configs["image_size"]
        self.batch_size = configs["batch_size"]
        self.workers = configs["workers"]
        pass

    def Get(self):

        if not os.path.exists(self.dataroot):
            os.makedirs(self.dataroot, exist_ok=True)
            
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset = dset.ImageFolder(root=self.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(self.image_size),
                                    transforms.CenterCrop(self.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=self.workers)
        return dataloader