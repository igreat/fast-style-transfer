import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from models import loss_models, transformation_models


class StyleModelTrainer:
    def __init__(self, model, loss_model, optimizer, training_config, device):
        self.transformation_model = model.to(device)
        self.loss_model = loss_model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.training_config = training_config
        self.summary = SummaryWriter()

    def get_training_loader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.training_config["img_size"]),
                transforms.CenterCrop(self.training_config["img_size"]),
                transforms.ToTensor(),
                transforms.Normalize(self.loss_model.MEAN, self.loss_model.STD),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root=self.training_config["path_to_dataset"], transform=transform
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_config["batch_size"], shuffle=True
        )
        return train_loader

    def train(self):
        train_loader = self.get_training_loader()

        # there is a bug when doing multiple epochs where the batch size for some reason becomes 3
        # (or something else)

        # training
        for epoch in range(self.training_config["epochs"]):
            size = len(train_loader.dataset)
            self.transformation_model.train()
            for batch, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                result = self.transformation_model(x)

                loss = self.loss_model(result, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # printing losses
                content_loss = self.loss_model.total_content_loss
                style_loss = self.loss_model.total_style_loss
                tv_loss = self.loss_model.tv_loss.loss
                current_iteration = batch * len(x)
                if current_iteration % 250 == 0:
                    current = batch * len(x)
                    print(f"style loss: {style_loss.item():>7f}", end="\t")
                    print(f"content loss: {content_loss.item():>7f}", end="\t")
                    print(f"tv loss: {tv_loss.item():>7f}", end="\t")
                    print(f"total loss: {loss.item():>7f}", end="\t")
                    print(f"[{current:>5d}/{size:>5d}]")

                # autosaving every 1500 training steps
                if current_iteration % 5e2 == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.transformation_model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": loss,
                        },
                        "auto_save.pth",
                    )

                # adding losses to tensorboard
                self.summary.add_scalar(
                    "losses/content",
                    content_loss.item(),
                    current_iteration + epoch * len(train_loader),
                )
                self.summary.add_scalar(
                    "losses/style",
                    style_loss.item(),
                    current_iteration + epoch * len(train_loader),
                )
                self.summary.add_scalar(
                    "losses/tv",
                    tv_loss.item(),
                    current_iteration + epoch * len(train_loader),
                )

                if current_iteration % 400 == 0:
                    # preparing and displaying the example image
                    example_image = x[0] * self.loss_model.STD + self.loss_model.STD
                    example_image = example_image.clamp(0, 1) * 255
                    example_image = (
                        example_image.detach().cpu().numpy().astype(np.uint8)
                    )
                    self.summary.add_image(
                        "images/example_image", example_image, current_iteration + 1
                    )
                    # preparing and displaying the styled image
                    example_result = (
                        result[0] * self.loss_model.STD + self.loss_model.STD
                    )
                    example_result = example_result.clamp(0, 1) * 255
                    example_result = (
                        example_result.detach().cpu().numpy().astype(np.uint8)
                    )
                    self.summary.add_image(
                        "images/example_styled_image",
                        example_result,
                        current_iteration + 1,
                    )

        torch.save(
            self.transformation_model.state_dict(), "saved-models/trained_model.pth"
        )

if __name__ == "__main__":
    # setting up the device
    device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")
    print(f"Using {device} device")

    # setting up the model
    transformation_model = transformation_models.TransformationModel()

    # setting up the loss model and optimizer
    style_img = (
        pil_to_tensor((Image.open("images/starry-night.jpg")).convert("RGB"))
        .unsqueeze(0)
        .float()
        .div(255)
    )
    mean, std = loss_models.VGG16Loss.MEAN, loss_models.VGG16Loss.STD
    style_img = (style_img - mean) / std

    loss_model = loss_models.VGG16Loss(style_img, device=device)
    optimizer = torch.optim.Adam(transformation_model.parameters(), lr=1e-3)

    # setting up the training config
    training_config = {
        "path_to_dataset": "data/mscoco",
        "batch_size": 4,
        "img_size": 256,
        "epochs": 1,
    }

    # training the model
    trainer = StyleModelTrainer(
        transformation_model, loss_model, optimizer, training_config, device
    )
    trainer.train()
    print("Training complete!")
    