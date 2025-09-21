import torch
from torchvision.transforms import v2 as T
from src.pytorch.datasets import PennFudanDataset


class RCnnDataLoader:
    def __init__(self, configs=None):
        self.config = configs if configs is not None else {}
        self.device = self.config["device"] if "device" in self.config else "cpu"
        return self
    
    def get_transform(train):
        transforms = []
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    def Get(self):
        self.config["num_classes"] = 2
        dataset = PennFudanDataset(self.config["data_root"], self.get_transform(train=True))
        dataset_test = PennFudanDataset(self.config["data_root"], self.get_transform(train=False))

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # define training and validation data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        return train_loader, test_loader