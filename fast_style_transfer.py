# TODO: separate the training and running of the model into two files
# TODO: think about changing the training dataset to 2014 COCO dataset instead of 2017

import torch
from torchvision import transforms
from models import loss_models, transformation_models
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from torch import optim
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor, resize

device = "mps"

transform = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ]
)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

normalize = transforms.Normalize(mean, std)


def train():

    train_dataset = datasets.ImageFolder(root="data/fiftyk", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    style_img = (
        pil_to_tensor((Image.open("images/mosaic.jpg")).convert("RGB"))
        .to(device)
        .unsqueeze(0)
        .float()
        .div(255)
    )
    style_img = normalize(style_img)
    loss_model = loss_models.VGG16Loss(style_img, device=device, batch_size=4)

    transformation_model = transformation_models.TransformationModel().to(device)

    optimizer = optim.Adam(transformation_model.parameters())

    losses = {"content": [], "style": []}

    # training for two epochs
    size = len(train_loader.dataset)
    transformation_model.train()
    for batch, (x, _) in enumerate(train_loader):
        x = x.to(device)  # experiment with different init methods
        result = transformation_model(x)

        x = x.div(255)
        result = result.div(255)

        x = normalize(x)
        result = normalize(result)

        loss = loss_model(result, x)

        # logging losses
        if losses:
            losses["content"].append(loss_model.total_content_loss.item())
            losses["style"].append(loss_model.total_style_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_iteration = batch * len(x)
        if current_iteration % 100 == 0:
            current = batch * len(x)
            print(f"style loss: {loss_model.total_style_loss.item():>7f}", end="\t")
            print(f"content loss: {loss_model.total_content_loss.item():>7f}", end="\t")
            print(f"total loss: {loss.item():>7f}", end="\t")
            print(f"[{current:>5d}/{size:>5d}]")

        # autosaving every 1000 training steps
        if current_iteration % 1e3 == 0:
            torch.save(transformation_model.state_dict(), "auto_save.pth")

    torch.save(transformation_model.state_dict(), "saved-models/trained_model.pth")


def apply_style(image, path_to_model):

    image = (
        pil_to_tensor((Image.open(image)).convert("RGB"))
        .to(device)
        .unsqueeze(0)
        .float()
        .div(255)
    )

    transformation_model = transformation_models.TransformationModel().to(device)

    # code to load pretrained model
    transformation_model.load_state_dict(
        torch.load(path_to_model, map_location=torch.device("cpu"))
    )

    gen_image = transformation_model.eval()(image).div(255)

    # saving image
    save_image(gen_image.squeeze(0), "styled_image.png")


def main():
    train()
    apply_style("images/monalisa.jpg", "saved-models/trained_model.pth")


if __name__ == "__main__":
    main()
