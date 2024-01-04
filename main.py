from typing import Any

import PIL
from PIL import Image
import numpy as np
import torch
import lightning as L
import torchvision as torchvision
from torch.utils.data import DataLoader,IterableDataset
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb


class LayerNormHelper(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.LayerNorm(x.shape[1:])(x)

class Generator(L.LightningModule):
    def __init__(self, depth=1, noise_size=(1,1), noise_channels=100):
        super().__init__()

        self.noise_size=noise_size
        self.noise_channels=noise_channels
        self.fire_power=2**depth

        # Input: 100x1x1

        # Dense layer
        self.model=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=noise_channels, out_channels=self.fire_power*8, kernel_size=4, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power*8,4,4]),
            torch.nn.ConvTranspose2d(in_channels=self.fire_power*8, out_channels=self.fire_power*4, kernel_size=4, stride=2,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power*4,8,8]),
            torch.nn.ConvTranspose2d(in_channels=self.fire_power*4, out_channels=self.fire_power*2, kernel_size=4, stride=2,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power*2,16,16]),
            torch.nn.ConvTranspose2d(in_channels=self.fire_power*2, out_channels=self.fire_power, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power,32,32]),
            torch.nn.ConvTranspose2d(in_channels=self.fire_power, out_channels=3, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
        )




    def forward(self, x) -> torch.Tensor:
        return self.model(x)

class Discriminator(torch.nn.Module):

    def __init__(self,depth=1, image_size=(32,32)):
        super().__init__()
        self.depth=depth
        self.image_size=image_size
        self.fire_power=2**depth

        self.model=torch.nn.Sequential(
            # 32x32x3
            torch.nn.Conv2d(in_channels=3, out_channels=self.fire_power, kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power,32,32]),
            # 16x16x64
            torch.nn.Conv2d(in_channels=self.fire_power, out_channels=self.fire_power*2, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power*2,16,16]),
            # 8x8x128
            torch.nn.Conv2d(in_channels=self.fire_power*2, out_channels=self.fire_power*4, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power*4,8,8]),
            # 4x4x256
            torch.nn.Conv2d(in_channels=self.fire_power*4, out_channels=self.fire_power*8, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([self.fire_power*8,4,4]),
            # 1x1x512
            torch.nn.Conv2d(in_channels=self.fire_power*8, out_channels=1, kernel_size=4, stride=1, padding=0),
            torch.nn.LeakyReLU()
        )



    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class ImprovedWassersteinGAN(L.LightningModule):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()
        self.generator=generator
        self.discriminator=discriminator
        self.automatic_optimization = False
        self.counter=0
        self.noise_size=lambda batch_size:[batch_size,self.generator.noise_channels,*self.generator.noise_size]
        # To generate a 10x10 grid of images
        self.test_noise=torch.randn(self.noise_size(100),device=self.device,requires_grad=False,dtype=self.dtype)


    def improved_wasserstein_loss(self,fake_scores, interpolated_scores,interpolated_images,lambda_=10):
        differentiation=torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated_images,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients=differentiation.view(differentiation.shape[0],-1)
        gradients=gradients.norm(2,dim=1)
        gradient_loss=lambda_ * torch.pow((gradients - 1),2).view(-1,1)

        return fake_scores-interpolated_scores+gradient_loss



    def training_step(self, batch, batch_idx):
        gan_optimizer, dis_optimizer = self.optimizers()

        # Create a uniform distribution of size 1 to signify if we should train the discriminator or generator





        for _ in range(10):
            # Create random noise for the generator at size batch
            noise=torch.randn(self.noise_size(batch_size=batch[0].shape[0]),device=self.device, dtype=self.dtype)
            # Generate fake images
            fake_images=self.generator(noise)
            # Discriminate fake images
            fake_scores=self.discriminator(fake_images)
            # Create alpha with size batch, 1
            alpha=torch.rand(batch[0].shape[0],1,1,1,device=self.device, dtype=self.dtype)
            # Create interpolated images
            interpolated_images=(alpha*batch[0]+(1-alpha)*fake_images).requires_grad_(True)
            # Discriminate the interpolated images
            interpolated_scores=self.discriminator(interpolated_images)
            # Calculate the loss
            dis_loss=self.improved_wasserstein_loss(
                fake_scores=fake_scores,
                interpolated_scores=interpolated_scores,
                interpolated_images=interpolated_images
            ).mean()

            # Update the discriminator
            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()
            self.logger.log_metrics({"dis_loss":dis_loss},self.global_step)
            self.log("dis_loss", dis_loss, True,False)


        noise = torch.randn(self.noise_size(batch_size=batch[0].shape[0]),device=self.device, dtype=self.dtype)
        fake_images = self.generator(noise)
        fake_scores = self.discriminator(fake_images)
        gen_loss = -torch.mean(fake_scores)
        gan_optimizer.zero_grad()
        gen_loss.backward()
        gan_optimizer.step()
        self.logger.log_metrics({"gen_loss": gen_loss}, self.global_step)
        self.log("gen_loss", gen_loss,True,False)


        if self.counter%10==0:
            with torch.no_grad():

                self.test_noise=self.test_noise.to(self.device)

                test_images=self.generator(self.test_noise)
                # Create a grid of images
                grid=torchvision.utils.make_grid(test_images,nrow=10)
                # Image to tensorboard
                self.logger.experiment.add_image("fake_image",grid,self.global_step)
        self.counter+=1

        # Schedule the learning rate
        gan_scheduler, dis_scheduler = self.lr_schedulers()
        gan_scheduler.step()
        dis_scheduler.step()





    def configure_optimizers(self) -> OptimizerLRScheduler:
        gan_optimizer = torch.optim.SGD(self.generator.parameters(), lr=1e-3, momentum=0.09)
        dis_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=1e-3, momentum=0.09)
        milestones=[100,400,1600,3200,6400]

        # Gan scheduler
        gan_scheduler = torch.optim.lr_scheduler.MultiStepLR(gan_optimizer, milestones=milestones, gamma=0.1)
        # Dis scheduler
        dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(dis_optimizer, milestones=milestones, gamma=0.1)


        return [gan_optimizer, dis_optimizer], [gan_scheduler, dis_scheduler]


if __name__ == "__main__":
    wandb.login()

    transforms=torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        lambda x: x/255.0,

        # torchvision.transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))

    ])

    tensorboard_logger = TensorBoardLogger('logs/')
    wandb_logger = WandbLogger(project="Improved Training of Wasserstein GANs")

    cifar100=torchvision.datasets.CIFAR100('./train/', download=True, train=True, transform=transforms)
    trainer = L.Trainer(logger=[
        tensorboard_logger,
        wandb_logger
    ],
        max_epochs=-1,
    )

    # Get one class from cifar100
    cifar100 = torch.utils.data.Subset(cifar100, [i for i in range(len(cifar100)) if cifar100[i][1] == 98])
    # Convert to Dataset
    cifar100 = torch.utils.data.TensorDataset(torch.stack([x[0] for x in cifar100]))
    # Convert to dataloader
    cifar100 = torch.utils.data.DataLoader(cifar100, batch_size=500, shuffle=True, num_workers=8, pin_memory=True)
    torch.set_float32_matmul_precision('medium')

    trainer.fit(ImprovedWassersteinGAN(Generator(depth=2), Discriminator(depth=3,image_size=(32,32))), cifar100)




