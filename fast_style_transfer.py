# TODO: make clean model saving functionality

from torchvision import transforms
from models import loss_models, transformation_models
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from torch import optim
from torchvision.utils import save_image

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


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)


def main():
    train_dataset = datasets.ImageFolder(root="data", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    style_img = transform(Image.open("images/starry-night.jpg")).to(device).unsqueeze(0)
    loss_model = loss_models.VGG19Loss(style_img, device=device)

    model = transformation_models.TransformationModel().to(device)

    optimizer = optim.Adam(model.parameters())

    # training for two epochs
    train(train_loader, model, loss_model, optimizer)
    train(train_loader, model, loss_model, optimizer)

    # testing it on a sample image
    test_image = transform(Image.open("images/houses.jpg")).to(device).unsqueeze(0)
    gen_image = model.eval()(test_image).div(255)

    # saving image
    save_image(gen_image.squeeze(0), "test.png")


if __name__ == "__main__":
    main()
