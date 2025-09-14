import torch
import torchvision.utils as vutils

class DCGANTrainer:
    def __init__(self, modelG, modelD, optimizerG, optimizerD, criterion, dataloader, device, configs):
        self.modelG = modelG
        self.modelD = modelD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.configs = configs
        # Training Loop
        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.iters = 0
        self.fixed_noise = torch.randn(64, configs["nz"], 1, 1, device=device)
        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

    def train(self, epoch):
        
        for i, data in enumerate(self.dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            self.modelD.zero_grad()
            # Format batch
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
            # Forward pass real batch through D
            output = self.modelD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.configs["nz"], 1, 1, device=self.device)
            # Generate fake image batch with G
            fake = self.modelG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = self.modelD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.modelG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.modelD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, self.configs["num_epochs"], i, len(self.dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == self.configs["num_epochs"]-1) and (i == len(self.dataloader)-1)):
                with torch.no_grad():
                    fake = self.modelG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1