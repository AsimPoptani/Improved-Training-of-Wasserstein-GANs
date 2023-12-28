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
# import wandb


class LayerNormHelper(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.LayerNorm(x.shape[1:])(x)

class Generator(L.LightningModule):
    def __init__(self, depth=1, noise_size=(8,8), noise_channels=1):
        super().__init__()
        self.noise_size=noise_size
        self.noise_channels=noise_channels
        self.depth=depth
        # 3x32x32 of gaussian noise
        self.initial=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=noise_channels, out_channels=64, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64)
        )
        # +2

        self.block = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64)
        ) for index in range(4)])
        # +8
        # Convert to 32x32x3
        self.final=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5),
            # +2
            torch.nn.LeakyReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        x=self.initial(x)
        for nn in self.block:
            x=nn(x)
        x=self.final(x)
        return x

class Discriminator(torch.nn.Module):

    def __init__(self,depth=1, image_size=(32,32)):
        super().__init__()
        self.depth=depth
        self.image_size=image_size
        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([64,*image_size])
        )

        self.block = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([64,*image_size])
        ) for _ in range(self.depth)])

        # 32x32x64 to 1
        self.final=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm([1,*image_size]),
            torch.nn.Flatten(),
            torch.nn.Linear(image_size[0]*image_size[1],1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x) -> torch.Tensor:
        x=self.initial(x)
        for nn in self.block:
            skip = x
            x = nn(x)
            x += skip
        x=self.final(x)
        return x



def torch_to_image(x: torch.tensor) -> PIL.Image:
    # Detach from graph and move to cpu
    x=x.detach().cpu()
    # Convert to 0 to 1
    x=(x-x.min())/(x.max()-x.min())

    # Reshape to 32x32x3
    x=x.reshape(32,32,3)

    # Convert to PIL image
    x=Image.fromarray(np.uint8(x.numpy()*255))
    return x







class ImprovedWassersteinGAN(L.LightningModule):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()
        self.generator=generator
        self.discriminator=discriminator
        self.automatic_optimization = False
        self.counter=0
        self.noise_size=lambda batch_size:[batch_size,self.generator.noise_channels,*self.generator.noise_size]


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
            # fake_image:PIL.Image=torchvision.transforms.ToPILImage(fake_images[0])
            # Image to tensorboard
            self.logger.experiment.add_image("fake_image",fake_images[0],self.global_step)



        self.counter+=1







    def configure_optimizers(self) -> OptimizerLRScheduler:
        gan_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-2)
        return [gan_optimizer, dis_optimizer], []


if __name__ == "__main__":
    # wandb.login()

    transforms=torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        lambda x: x/255.0,

        # torchvision.transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))

    ])

    tensorboard_logger = TensorBoardLogger('logs/')
    wandb_logger = WandbLogger(project="Improved Training of Wasserstein GANs", offline=True)

    cifar100=torchvision.datasets.CIFAR100('./train/', download=True, train=True, transform=transforms)
    trainer = L.Trainer(logger=[
        tensorboard_logger,
        wandb_logger
    ])
    # Get one class from cifar100
    cifar100 = torch.utils.data.Subset(cifar100, [i for i in range(len(cifar100)) if cifar100[i][1] == 98])
    # Convert to Dataset
    cifar100 = torch.utils.data.TensorDataset(torch.stack([x[0] for x in cifar100]))
    # Convert to dataloader
    cifar100 = torch.utils.data.DataLoader(cifar100, batch_size=100, shuffle=True, num_workers=9)


    trainer.fit(ImprovedWassersteinGAN(Generator(depth=10), Discriminator(depth=10,image_size=(32,32))), cifar100)




