# TODO: separate the training and running of the model into two files

import torch
import numpy as np
from torchvision import transforms
from models import loss_models, transformation_models
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from torch import optim
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor, resize
from torch.utils.tensorboard import SummaryWriter
from utils import display_images_in_a_grid

device = "mps" if torch.has_mps else "cpu"

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

img_size = 256
transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


def train():
    summary = SummaryWriter()

    train_dataset = datasets.ImageFolder(root="data/mscoco", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    style_img = (
        pil_to_tensor((Image.open("images/starry-night.jpg")).convert("RGB"))
        .to(device)
        .unsqueeze(0)
        .float()
        .div(255)
    )
    style_img = (style_img - mean) / std
    loss_model = loss_models.VGG16Loss(style_img, device=device, batch_size=4)

    transformation_model = transformation_models.TransformationModel().to(device)

    optimizer = optim.Adam(transformation_model.parameters())

    # load model and optimizer
    checkpoint = torch.load("auto_save.pth")
    transformation_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # there is a bug when doing multiple epochs where the batch size for some reason becomes 3
    # training for one epochs
    for epoch in range(1):
        size = len(train_loader.dataset)
        transformation_model.train()
        for batch, (x, _) in enumerate(train_loader):
            x = x.to(device)
            result = transformation_model(x)

            loss = loss_model(result, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # printing losses
            content_loss = loss_model.total_content_loss
            style_loss = loss_model.total_style_loss
            # tv_loss = loss_model.tv_loss.loss
            current_iteration = batch * len(x)
            if current_iteration % 100 == 0:
                current = batch * len(x)
                print(f"style loss: {style_loss.item():>7f}", end="\t")
                print(f"content loss: {content_loss.item():>7f}", end="\t")
                print(f"total loss: {loss.item():>7f}", end="\t")
                print(f"[{current:>5d}/{size:>5d}]")

            # autosaving every 1000 training steps
            if current_iteration % 1e3 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": transformation_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    "auto_save.pth",
                )

                # torch.save(transformation_model.state_dict(), "")

            summary.add_scalar(
                "losses/content",
                content_loss.item(),
                current_iteration + epoch * len(train_loader),
            )
            summary.add_scalar(
                "losses/style",
                style_loss.item(),
                current_iteration + epoch * len(train_loader),
            )

            if current_iteration % 400 == 0:
                # preparing and displaying the example image
                example_image = x[0] * std + mean
                example_image = example_image.clamp(0, 1) * 255
                example_image = example_image.detach().cpu().numpy().astype(np.uint8)
                summary.add_image(
                    "images/example_image", example_image, current_iteration + 1
                )
                # preparing and displaying the styled image
                example_result = result[0] * std + mean
                example_result = example_result.clamp(0, 1) * 255
                example_result = example_result.detach().cpu().numpy().astype(np.uint8)
                summary.add_image(
                    "images/example_styled_image", example_result, current_iteration + 1
                )

    torch.save(transformation_model.state_dict(), "saved-models/trained_model.pth")


def apply_style(path_to_image, path_to_model):

    img = (
        pil_to_tensor((Image.open(path_to_image)).convert("RGB"))
        .to(device)
        .unsqueeze(0)
        .float()
        .div(255)
    )

    transformation_model = transformation_models.TransformationModel().to(device)

    # code to load pretrained model
    checkpoint = torch.load(path_to_model)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])

    gen_image = transformation_model.eval()(img)
    gen_image = gen_image * std + mean
    gen_image = gen_image.clamp(0, 1)

    # saving image
    save_image(gen_image.squeeze(0), "styled_image.png")


def apply_style_grid(path_to_image, paths_to_models):
    # produces a grid of images in matplotlib for the outputs of multiple models on the same image
    # I used this to compare multiple checkpoints of the same model

    img = resize(
        pil_to_tensor((Image.open(path_to_image)).convert("RGB"))
        .to(device)
        .unsqueeze(0)
        .float()
        .div(255),
        512,
    )
    transformation_model = transformation_models.TransformationModel().to(device)

    # code to load pretrained models
    models = []
    for path in paths_to_models:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        transformation_model.load_state_dict(checkpoint["model_state_dict"])
        models.append(transformation_model.eval())

    gen_images = []
    for model in models:
        gen_image = model(img)
        gen_image = gen_image * std + mean
        gen_image = gen_image.clamp(0, 1)
        gen_image = gen_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        gen_images.append(gen_image)

    # display images in a grid
    display_images_in_a_grid(gen_images, 4, paths_to_models)


def main():
    # train()
    apply_style(
        "images/me.jpg",
        "saved-models/starry_night_pretrained.pth",
    )


if __name__ == "__main__":
    main()
