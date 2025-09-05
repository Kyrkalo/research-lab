import torch
import torch.nn.functional as F
import os
from pathlib import Path

class MNISTTrainer:
    def __init__(self, model, optimizer, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.log_interval = config["log_interval"]
        self.train_losses = []
        self.train_counter = []
        self.device = config["device"]
        self.model_name = config["model_name"] if "model_name" in config else "mnist_model"
        self.optimizer_name = config["optimizer_name"] if "optimizer_name" in config else "mnist_optimizer"

    def train(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)

            loss = F.cross_entropy(output, target)  # (optional: label_smoothing=0.1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # optional safety
            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                seen = batch_idx * len(data)
                pct = 100.0 * batch_idx / len(self.train_loader)
                print(f"Train Epoch: {epoch} [{seen}/{len(self.train_loader.dataset)} ({pct:.0f}%)]\t"
                      f"Loss: {loss.item():.6f}")
                self.train_losses.append(loss.item())
                
                self.train_counter.append(seen + ((epoch - 1) * len(self.train_loader.dataset)))

                torch.save(self.model.state_dict(), self.model_name + ".pth")
                torch.save(self.optimizer.state_dict(), self.optimizer_name + ".pth")

        avg_loss = running_loss / len(self.train_loader)
        print(f"===> Epoch {epoch} Average training loss: {avg_loss:.6f}")

    def get_train_stats(self):
        return self.train_losses, self.train_counter