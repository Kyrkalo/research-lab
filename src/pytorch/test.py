import torch
import torch.nn.functional as F

class MNISTTester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.test_losses = []
        self.device = device

    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.test_loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            output = self.model(data)
            # Match training loss: CrossEntropy on logits
            loss = F.cross_entropy(output, target, reduction="sum")
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        print(f"\nTest Epoch: {epoch}  Average loss: {avg_loss:.4f}  Accuracy: {correct}/{total} ({acc:.2f}%)\n")
        
    def get_test_stats(self):
        return self.test_losses, self.test_counter