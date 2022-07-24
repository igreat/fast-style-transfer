# TODO: make clean model saving functionality
# TODO: Experiment with contatenating noise with input image
#       Think about adding this process as part of the forward pass
#       of the transformation network

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


def train(data_loader, model, loss_model, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, _) in enumerate(data_loader):
        X = X.to(device)  # experiment with different init methods
        result = model(X)
        loss = loss_model(result, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch * len(X) % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # autosaving every 2000 training steps
        if batch * len(X) % 2e3 == 0:
            torch.save(model.state_dict(), "auto_save.pth")


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)


def main():
    train_dataset = datasets.ImageFolder(root="data/fiftyk", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    style_img = transform(Image.open("images/starry-night.jpg")).to(device).unsqueeze(0)
    loss_model = loss_models.VGG19Loss(style_img, device=device)

    transformation_model = transformation_models.TransformationModel().to(device)

    # code to load pretrained model
    transformation_model.load_state_dict(
        torch.load("saved-models/test_model_IN.pth", map_location=torch.device("cpu"))
    )

    optimizer = optim.Adam(transformation_model.parameters())

    # training for one epoch
    train(train_loader, transformation_model, loss_model, optimizer)

    torch.save(transformation_model.state_dict(), "saved-models/in_test_model.pth")

    # testing it on a sample image
    test_image = resize(
        pil_to_tensor(Image.open("images/sultan-qaboos-grand-mosque.jpg"))
        .mul(255.0)
        .to(device)
        .unsqueeze(0),
        400,
    )
    gen_image = transformation_model.eval()(test_image).div(255)

    # saving image
    save_image(gen_image.squeeze(0), "test.png")


if __name__ == "__main__":
    main()
